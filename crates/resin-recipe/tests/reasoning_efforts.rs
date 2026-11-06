use resin_recipe::{
    Recipe,
    deepseek::DeepSeekV41Recipe,
    glm::Glm53FlashRecipe,
    mistral::MistralLarge4Recipe,
    qwen::{Qwen35Recipe, Qwen38Recipe},
};
use serde_json::{Value, json};

fn request(effort: Option<&str>) -> Value {
    let mut request = json!({"messages": [{"role": "user", "content": "hi"}], "tools": []});
    if let Some(effort) = effort {
        request["reasoning_effort"] = json!(effort);
    }
    request
}

fn rendered(recipe: &dyn Recipe, effort: Option<&str>) -> Option<String> {
    recipe
        .render(&request(effort))
        .ok()
        .map(|prompt| prompt.rendered_text())
}

/// Each listed effort renders a different prompt, and an effort outside the list fails or
/// renders as a listed effort does. Thus the list holds every level that the recipe has.
#[test]
fn listed_reasoning_efforts_are_the_levels_of_each_recipe() {
    let recipes: [(&str, Box<dyn Recipe>, &[&str]); 4] = [
        (
            "qwen3_8",
            Box::new(Qwen38Recipe::new(Default::default())),
            &["low", "medium", "xhigh"],
        ),
        (
            "glm5_3_flash",
            Box::new(Glm53FlashRecipe::new(Default::default())),
            &["low", "high", "max"],
        ),
        (
            "deepseek_v41",
            Box::new(DeepSeekV41Recipe::new(Default::default()).unwrap()),
            &["low", "high", "max"],
        ),
        (
            "mistral_large4",
            Box::new(MistralLarge4Recipe::new(Default::default())),
            &["low", "medium", "high"],
        ),
    ];
    for (name, recipe, expected) in recipes {
        assert_eq!(recipe.reasoning_efforts(), expected, "{name}");
        let prompts = expected
            .iter()
            .map(|effort| {
                rendered(recipe.as_ref(), Some(effort))
                    .unwrap_or_else(|| panic!("{name} refuses {effort}"))
            })
            .collect::<Vec<_>>();
        for (index, prompt) in prompts.iter().enumerate() {
            assert!(
                !prompts[..index].contains(prompt),
                "{name}: {} renders as an earlier effort",
                expected[index]
            );
        }
        for other in ["none", "minimal", "medium", "high", "xhigh", "max", "bogus"] {
            if expected.contains(&other) {
                continue;
            }
            if let Some(prompt) = rendered(recipe.as_ref(), Some(other)) {
                assert!(
                    prompts.contains(&prompt),
                    "{name}: {other} renders a prompt that no listed effort renders"
                );
            }
        }
    }
}

#[test]
fn qwen3_5_lists_no_reasoning_efforts_because_it_ignores_them() {
    let recipe = Qwen35Recipe::new(Default::default());
    assert!(recipe.reasoning_efforts().is_empty());
    let default = rendered(&recipe, None).unwrap();
    for effort in ["low", "medium", "xhigh"] {
        assert_eq!(rendered(&recipe, Some(effort)).as_ref(), Some(&default));
    }
}
