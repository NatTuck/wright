use resin_recipe::{
    AttributedText, Decoder, Error, Event, FinishReason, MistralParser, MistralTokens, Recipe,
    mistral::{MistralLarge4Recipe, MistralLarge4RecipeParams},
};
use serde_json::{Value, json};
mod common;
use common::SemanticParser;

const TOKENS: MistralTokens = MistralTokens {
    eos: 1000,
    think: 1001,
    think_end: 1002,
    tool_calls: 1003,
    args: 1004,
    call_id: 1005,
};
#[derive(Default)]
struct Bytes(Vec<u8>);
impl Decoder for Bytes {
    fn push(&mut self, ids: &[u32], _token_start: u32) -> Result<AttributedText, Error> {
        self.0
            .extend(ids.iter().map(|&id| u8::try_from(id).unwrap()));
        let length = match std::str::from_utf8(&self.0) {
            Ok(_) => self.0.len(),
            Err(error) if error.error_len().is_none() => error.valid_up_to(),
            Err(_) => return Err(Error::new("invalid bytes")),
        };
        Ok(AttributedText::plain(
            String::from_utf8(self.0.drain(..length).collect()).unwrap(),
        ))
    }
    fn finish(&mut self) -> Result<AttributedText, Error> {
        let text = String::from_utf8_lossy(&self.0).into_owned();
        self.0.clear();
        Ok(AttributedText::plain(text))
    }
    fn reset(&mut self) {
        self.0.clear();
    }
}
fn bytes(text: &str) -> Vec<u32> {
    text.bytes().map(u32::from).collect()
}
fn parser() -> MistralParser {
    MistralParser::new(TOKENS, Box::<Bytes>::default(), &[4]).unwrap()
}
fn coalesce(events: Vec<Event>) -> Vec<Event> {
    let mut out = Vec::new();
    for event in events {
        match (out.last_mut(), event) {
            (Some(Event::TextDelta { delta }), Event::TextDelta { delta: next })
            | (Some(Event::ReasoningDelta { delta }), Event::ReasoningDelta { delta: next }) => {
                delta.push_str(&next)
            }
            (_, event) => out.push(event),
        }
    }
    out
}

#[test]
fn pinned_hf_template_matches_reasoning_tools_and_merging() {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/mistral_hf.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let actual = MistralLarge4Recipe::default()
            .render(&case["request"])
            .unwrap();
        assert_eq!(
            actual.rendered_text(),
            case["text"].as_str().unwrap(),
            "{}",
            case["name"]
        );
    }
}

#[test]
fn efforts_map_and_invalid_conversations_fail() {
    let recipe = MistralLarge4Recipe::new(MistralLarge4RecipeParams {
        reasoning_effort: Some("xhigh".into()),
    });
    let request = json!({"messages":[{"role":"user","content":"hi"}]});
    let text = recipe.render(&request).unwrap().rendered_text();
    assert!(text.contains(r#"{"reasoning_effort": "high"}"#));
    let mut other = request.clone();
    other["extra"] = json!({"reasoning_effort":"low"});
    assert!(
        recipe
            .render(&other)
            .unwrap()
            .rendered_text()
            .contains("\"low\"")
    );
    other["extra"] = json!({"reasoning_effort":"none"});
    assert!(recipe.render(&other).is_err());
    other["extra"] = json!({"enable_thinking":false});
    assert!(recipe.render(&other).is_err());
    for messages in [
        json!([{"role":"assistant","content":"x"}]),
        json!([{"role":"user","content":""}]),
        json!([{"role":"user","content":"a"},{"role":"tool","content":"b"}]),
        json!([{"role":"user","content":"a"},{"role":"assistant","content":null}]),
        json!([{"role":"user","content":[{"type":"image_url","image_url":{"url":"x"}}]}]),
    ] {
        assert!(
            recipe.render(&json!({"messages": messages})).is_err(),
            "{messages}"
        );
    }
}

#[test]
fn reasoning_text_and_tools_are_invariant_to_every_chunk_width() {
    let mut ids = vec![TOKENS.think];
    ids.extend(bytes("think 🦀\n"));
    ids.push(TOKENS.think_end);
    ids.extend(bytes("answer\n"));
    ids.push(TOKENS.tool_calls);
    ids.extend(bytes("get_weather"));
    ids.push(TOKENS.args);
    ids.extend(bytes(r#"{"city": "Paris"}"#));
    ids.push(TOKENS.tool_calls);
    ids.extend(bytes("f"));
    ids.push(TOKENS.call_id);
    ids.extend(bytes("abc"));
    ids.push(TOKENS.args);
    ids.extend(bytes(" {} "));
    ids.push(TOKENS.eos);
    let expected = vec![
        Event::ReasoningDelta {
            delta: "think 🦀\n".into(),
        },
        Event::TextDelta {
            delta: "answer\n".into(),
        },
        Event::ToolCallStart {
            index: 0,
            id: "call_0".into(),
            name: "get_weather".into(),
        },
        Event::ToolCallArgumentsDelta {
            index: 0,
            delta: r#"{"city": "Paris"}"#.into(),
        },
        Event::ToolCallStart {
            index: 1,
            id: "call_1".into(),
            name: "f".into(),
        },
        Event::ToolCallArgumentsDelta {
            index: 1,
            delta: "{}".into(),
        },
    ];
    for width in 1..=ids.len() {
        let mut parser = parser();
        let mut events = Vec::new();
        for chunk in ids.chunks(width) {
            events.extend(parser.semantic_feed(chunk).unwrap());
        }
        events.extend(parser.semantic_finish(FinishReason::Stop).unwrap());
        assert_eq!(coalesce(events), expected, "width {width}");
    }
}

#[test]
fn a_call_cut_off_by_the_length_limit_is_still_emitted() {
    let mut parser = parser();
    let mut ids = vec![TOKENS.tool_calls];
    ids.extend(bytes("g"));
    ids.push(TOKENS.args);
    ids.extend(bytes(r#"{"a": 1"#));
    let mut events = parser.semantic_feed(&ids).unwrap();
    events.extend(parser.semantic_finish(FinishReason::Length).unwrap());
    assert_eq!(
        events,
        vec![
            Event::ToolCallStart {
                index: 0,
                id: "call_0".into(),
                name: "g".into()
            },
            Event::ToolCallArgumentsDelta {
                index: 0,
                delta: r#"{"a": 1"#.into()
            },
        ]
    );
}
