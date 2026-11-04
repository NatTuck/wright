//! Mistral-Large-4 format, following the release's `chat_template.jinja`
//! (mistralai/Mistral-Large-4-1T-A52B-NVFP4; the same template is embedded in the
//! unsloth GGUF). Text only: images are refused.

mod parser;
pub use parser::{MistralParser, MistralTokens};

use crate::{
    Error, Prompt,
    render::{function, list, messages, option, truthy, whitespace},
};
use serde_json::{Value, json};

/// `reasoning_effort` is the template's model setting (low, medium or high; the API's
/// stronger efforts map to high).
#[derive(Clone, Debug, Default, serde::Deserialize, serde::Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct MistralLarge4RecipeParams {
    pub reasoning_effort: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub struct MistralLarge4Recipe {
    params: MistralLarge4RecipeParams,
}
impl MistralLarge4Recipe {
    pub fn new(params: MistralLarge4RecipeParams) -> Self {
        Self { params }
    }
}

const MODEL: &str = "Mistral-Large-4";

fn effort(request: &Value) -> Result<&'static str, Error> {
    match option(request, "reasoning_effort").and_then(Value::as_str) {
        None | Some("high" | "xhigh" | "max") => Ok("high"),
        Some("medium") => Ok("medium"),
        Some("low" | "minimal") => Ok("low"),
        Some(other) => Err(Error::new(format!(
            "{MODEL} reasoning_effort must be low, medium or high, got {other}"
        ))),
    }
}

/// One aggregated message: the template merges consecutive messages of a role (other than
/// system and tool), joining their text with blank lines.
struct Merged {
    role: String,
    content: Value,
    tool_calls: Vec<Value>,
}

fn reasoning_of(message: &Value) -> Option<&str> {
    let field = message
        .get("reasoning_content")
        .filter(|value| !value.is_null())
        .or_else(|| message.get("reasoning"));
    field
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
}

fn merge(group: &[&Value]) -> Result<Merged, Error> {
    let mut text_parts: Vec<String> = Vec::new();
    let mut chunks: Vec<Value> = Vec::new();
    let mut non_text = false;
    let mut tool_calls = Vec::new();
    for message in group {
        let mut content = message.get("content").cloned().unwrap_or(Value::Null);
        if let Some(reasoning) = reasoning_of(message) {
            if list(&content)
                .iter()
                .any(|block| block["type"] == "thinking")
            {
                return Err(Error::new(
                    "Message cannot have both thinking chunks in content and a top-level `reasoning` or `reasoning_content` field.",
                ));
            }
            let think = json!({"type": "thinking", "thinking": reasoning});
            content = match &content {
                Value::String(text) if !text.is_empty() => {
                    json!([think, {"type": "text", "text": text}])
                }
                Value::Array(blocks) if !blocks.is_empty() => Value::Array(
                    std::iter::once(think)
                        .chain(blocks.iter().cloned())
                        .collect(),
                ),
                _ => json!([think]),
            };
        }
        match &content {
            Value::String(text) => text_parts.push(text.clone()),
            Value::Null => {}
            Value::Array(blocks) => {
                let mut own: Vec<String> = Vec::new();
                for block in blocks {
                    if block["type"] == "text" {
                        own.push(
                            block["text"]
                                .as_str()
                                .ok_or_else(|| Error::new("text content must be a string"))?
                                .to_owned(),
                        );
                        continue;
                    }
                    if !own.is_empty() {
                        text_parts.push(own.concat());
                        own.clear();
                    }
                    if !text_parts.is_empty() {
                        chunks.push(json!({"type": "text", "text": text_parts.join("\n\n")}));
                        text_parts.clear();
                    }
                    chunks.push(block.clone());
                    non_text = true;
                }
                if !own.is_empty() {
                    text_parts.push(own.concat());
                }
            }
            _ => return Err(Error::new("message content must be text or content blocks")),
        }
        tool_calls.extend(list(&message["tool_calls"]).iter().cloned());
    }
    let content = if non_text {
        if !text_parts.is_empty() {
            chunks.push(json!({"type": "text", "text": text_parts.join("\n\n")}));
        }
        Value::Array(chunks)
    } else {
        Value::String(text_parts.join("\n\n"))
    };
    Ok(Merged {
        role: group[0]["role"].as_str().unwrap_or("").to_owned(),
        content,
        tool_calls,
    })
}

fn aggregate(messages: &[Value]) -> Result<Vec<Merged>, Error> {
    let mut out = Vec::new();
    let mut index = 0;
    while index < messages.len() {
        let role = messages[index]["role"].as_str().unwrap_or("");
        let mut end = index + 1;
        if role != "system" && role != "tool" {
            while end < messages.len() && messages[end]["role"] == messages[index]["role"] {
                end += 1;
            }
        }
        let group: Vec<&Value> = messages[index..end].iter().collect();
        if role == "tool" {
            // tool messages pass through unmerged
            out.push(Merged {
                role: role.to_owned(),
                content: group[0].get("content").cloned().unwrap_or(Value::Null),
                tool_calls: Vec::new(),
            });
        } else {
            out.push(merge(&group)?);
        }
        index = end;
    }
    Ok(out)
}

fn validate_order(messages: &[Merged]) -> Result<(), Error> {
    if let Some(first) = messages.first()
        && first.role != "user"
        && first.role != "system"
    {
        return Err(Error::new(format!(
            "Conversation must start with a user or system message, got {}.",
            first.role
        )));
    }
    for pair in messages.windows(2) {
        let (previous, current) = (pair[0].role.as_str(), pair[1].role.as_str());
        let allowed: &[&str] = match previous {
            "system" => &["user", "assistant", "system"],
            "user" => &["assistant", "system", "user"],
            "assistant" => &["assistant", "user", "tool"],
            "tool" => &["assistant", "tool", "user"],
            _ => &[],
        };
        if !allowed.contains(&current) {
            return Err(Error::new(format!(
                "Unexpected role '{current}' after role '{previous}'"
            )));
        }
    }
    Ok(())
}

/// `render_content`: text, thinking (assistant turns) and image blocks (refused: text only).
fn content(
    prompt: &mut Prompt,
    value: &Value,
    context: &str,
    supported: &str,
    thinking: bool,
) -> Result<(), Error> {
    match value {
        Value::String(text) if !text.is_empty() => {
            prompt.text(text);
        }
        Value::Array(blocks) if !blocks.is_empty() => {
            for block in blocks {
                match block["type"].as_str() {
                    Some("text") => {
                        prompt.text(
                            block["text"]
                                .as_str()
                                .ok_or_else(|| Error::new("text content must be a string"))?,
                        );
                    }
                    Some("thinking") if thinking => {
                        prompt
                            .special("[THINK]")
                            .text(block["thinking"].as_str().unwrap_or(""));
                        if block.get("closed").is_none_or(truthy) {
                            prompt.special("[/THINK]");
                        }
                    }
                    Some("image" | "image_url") if supported.contains("image") => {
                        return Err(Error::new(format!("{MODEL} on this runner is text only")));
                    }
                    _ => {
                        return Err(Error::new(format!(
                            "Only {supported} chunks are supported in {context}."
                        )));
                    }
                }
            }
        }
        _ => {
            return Err(Error::new(format!(
                "{context} must have non-empty content."
            )));
        }
    }
    Ok(())
}

fn tool_call(prompt: &mut Prompt, call: &Value) -> Result<(), Error> {
    let call = function(call);
    let name = call["name"]
        .as_str()
        .ok_or_else(|| Error::new(format!("{MODEL} tool call requires a name")))?;
    let arguments = match &call["arguments"] {
        Value::String(text) if text.is_empty() => "{}".to_owned(),
        Value::String(text) => text.clone(),
        other => crate::json::spaced(other),
    };
    prompt
        .special("[TOOL_CALLS]")
        .text(name)
        .special("[ARGS]")
        .text(arguments);
    Ok(())
}

impl crate::Recipe for MistralLarge4Recipe {
    fn message_policy(&self) -> crate::MessagePolicy {
        crate::MessagePolicy {
            roles: &["system", "user", "assistant", "tool"],
            ..crate::MessagePolicy::STANDARD
        }
    }
    fn render(&self, request: &Value) -> Result<Prompt, Error> {
        let request = crate::recipe::request(MODEL, &self.params, request)?;
        let effort = effort(&request)?;
        let merged = aggregate(messages(&request, MODEL)?)?;
        validate_order(&merged)?;
        let tools = list(&request["tools"]);
        let mut prompt = Prompt::new();
        prompt.special("<s>");
        let mut settings_emitted = false;
        for message in &merged {
            match message.role.as_str() {
                "user" => {
                    if !settings_emitted {
                        if !tools.is_empty() {
                            prompt
                                .special("[AVAILABLE_TOOLS]")
                                .text(crate::json::spaced(&Value::Array(tools.to_vec())))
                                .special("[/AVAILABLE_TOOLS]");
                        }
                        prompt
                            .special("[MODEL_SETTINGS]")
                            .text(format!("{{\"reasoning_effort\": \"{effort}\"}}"))
                            .special("[/MODEL_SETTINGS]");
                        settings_emitted = true;
                    }
                    let mut value = message.content.clone();
                    // one image and one text block: the image first
                    if let Value::Array(blocks) = &value
                        && blocks.len() == 2
                        && blocks[0]["type"] == "text"
                        && matches!(blocks[1]["type"].as_str(), Some("image" | "image_url"))
                    {
                        value = Value::Array(vec![blocks[1].clone(), blocks[0].clone()]);
                    }
                    prompt.special("[INST]");
                    content(
                        &mut prompt,
                        &value,
                        "user message content",
                        "text, image and image_url",
                        false,
                    )?;
                    prompt.special("[/INST]");
                }
                "assistant" => {
                    if !truthy(&message.content) && message.tool_calls.is_empty() {
                        return Err(Error::new(
                            "Assistant message must have a string or a list of chunks in content or a list of tool calls.",
                        ));
                    }
                    if truthy(&message.content) {
                        content(
                            &mut prompt,
                            &message.content,
                            "assistant message contents",
                            "text and thinking",
                            true,
                        )?;
                    }
                    for call in &message.tool_calls {
                        tool_call(&mut prompt, call)?;
                    }
                    prompt.special("</s>");
                }
                "tool" => {
                    prompt.special("[TOOL_RESULTS]");
                    content(
                        &mut prompt,
                        &message.content,
                        "tool message contents",
                        "text and image",
                        false,
                    )?;
                    prompt.special("[/TOOL_RESULTS]");
                }
                "system" => {
                    prompt.special("[SYSTEM_PROMPT]");
                    content(
                        &mut prompt,
                        &message.content,
                        "system message contents",
                        "text",
                        false,
                    )?;
                    prompt.special("[/SYSTEM_PROMPT]");
                }
                other => {
                    return Err(Error::new(format!(
                        "Only user, assistant, system and tool roles are supported, got {other}."
                    )));
                }
            }
        }
        Ok(prompt)
    }
    fn create_parser(
        &self,
        encoder: &dyn crate::Encoder,
        decoder: Box<dyn crate::Decoder>,
        _request: &Value,
        prompt_ids: &[u32],
    ) -> Result<Box<dyn crate::ResponseParser>, Error> {
        Ok(Box::new(MistralParser::new(
            MistralTokens::resolve(encoder)?,
            decoder,
            prompt_ids,
        )?))
    }
    fn stop_token_ids(&self, encoder: &dyn crate::Encoder) -> Result<Vec<u32>, Error> {
        Ok(vec![encoder.special_id("</s>")?])
    }
}
