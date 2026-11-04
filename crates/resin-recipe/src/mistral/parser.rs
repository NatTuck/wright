use crate::parser::{EventSink, Grammar, Parser, emit_tool, tokens};
use crate::{AttributedText, Decoder, Error, FinishReason};

tokens! { MistralTokens {
    eos => ["</s>"], think => ["[THINK]"], think_end => ["[/THINK]"],
    tool_calls => ["[TOOL_CALLS]"], args => ["[ARGS]"], call_id => ["[CALL_ID]"],
}}

#[derive(Clone, Copy, PartialEq)]
enum State {
    Text,
    Reasoning,
    Name,
    CallId,
    Arguments,
}

/// Mistral reasoning (`[THINK]` .. `[/THINK]`) and tool calls
/// (`[TOOL_CALLS]name[ARGS]{json}`, repeated; an optional `[CALL_ID]id` before `[ARGS]` is
/// skipped), driven by actual control-token IDs. A call is emitted when the next call, the
/// end of the response or EOS closes its arguments.
pub type MistralParser = Parser<MistralGrammar>;
pub struct MistralGrammar {
    tokens: MistralTokens,
    state: State,
    name: String,
    arguments: String,
    tool_index: usize,
    ended: bool,
}
impl MistralParser {
    pub fn new(
        tokens: MistralTokens,
        decoder: Box<dyn Decoder>,
        prompt_ids: &[u32],
    ) -> Result<Self, Error> {
        tokens.validate()?;
        let state = if prompt_ids.last() == Some(&tokens.think) {
            State::Reasoning
        } else {
            State::Text
        };
        Ok(Self::with_grammar(
            MistralGrammar {
                tokens,
                state,
                name: String::new(),
                arguments: String::new(),
                tool_index: 0,
                ended: false,
            },
            decoder,
        ))
    }
}
impl MistralGrammar {
    fn close_call(&mut self, events: &mut EventSink) {
        if self.state == State::Arguments {
            let name = self.name.trim_matches(super::whitespace);
            if !name.is_empty() {
                let arguments = self.arguments.trim_matches(super::whitespace).to_owned();
                emit_tool(&mut self.tool_index, name, arguments, events);
            }
        }
        self.name.clear();
        self.arguments.clear();
    }
}
impl Grammar for MistralGrammar {
    fn channel(&self) -> Option<&'static str> {
        match self.state {
            State::Text => Some("content"),
            State::Reasoning => Some("reasoning"),
            _ => None,
        }
    }
    fn is_control(&self, id: u32) -> bool {
        self.tokens.ids().contains(&id)
    }
    fn ended(&self) -> bool {
        self.ended
    }
    fn text(&mut self, text: AttributedText, events: &mut EventSink) -> Result<(), Error> {
        if text.is_empty() {
            return Ok(());
        }
        match self.state {
            State::Text => events.text(text, false),
            State::Reasoning => events.text(text, true),
            State::Name => self.name.push_str(&text),
            State::Arguments => self.arguments.push_str(&text),
            State::CallId => {}
        }
        Ok(())
    }
    fn control(&mut self, id: u32, events: &mut EventSink) -> Result<(), Error> {
        let t = self.tokens;
        if id == t.eos {
            self.close_call(events);
            self.ended = true;
        } else if id == t.tool_calls {
            self.close_call(events);
            self.state = State::Name;
        } else if id == t.call_id && self.state == State::Name {
            self.state = State::CallId;
        } else if id == t.args && matches!(self.state, State::Name | State::CallId) {
            self.state = State::Arguments;
        } else if id == t.think && matches!(self.state, State::Text | State::Reasoning) {
            self.state = State::Reasoning;
        } else if id == t.think_end && matches!(self.state, State::Text | State::Reasoning) {
            self.state = State::Text;
        }
        Ok(())
    }
    fn finish(&mut self, _reason: FinishReason, events: &mut EventSink) -> Result<(), Error> {
        self.close_call(events);
        Ok(())
    }
}
