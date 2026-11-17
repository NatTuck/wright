//! Card time shared between the models of one process.
//!
//! Each model has its own Engine, and the Engines step one at a time on the same card.
//! `ModelShare` picks the next Engine: of the models with work, the one with the least step
//! time per unit of weight. A model banks no credit while it has no work: when it has work
//! again, it starts level with the least served model that kept working, so it cannot hold
//! the card for a long burst to catch up.

use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ModelShareError;

impl fmt::Display for ModelShareError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("model share weights must be one or more finite positive numbers")
    }
}

impl std::error::Error for ModelShareError {}

#[derive(Clone, Debug)]
pub struct ModelShare {
    /// Step seconds of each model, adjusted when it comes back from idle.
    used: Vec<f64>,
    weights: Vec<f64>,
    /// Which models had work at the last choice.
    ready: Vec<bool>,
}

impl ModelShare {
    pub fn new(weights: &[f64]) -> Result<Self, ModelShareError> {
        if weights.is_empty()
            || weights
                .iter()
                .any(|weight| !weight.is_finite() || *weight <= 0.0)
        {
            return Err(ModelShareError);
        }
        Ok(Self {
            used: vec![0.0; weights.len()],
            weights: weights.to_vec(),
            ready: vec![false; weights.len()],
        })
    }

    pub fn models(&self) -> usize {
        self.weights.len()
    }

    fn level(&self, model: usize) -> f64 {
        self.used[model] / self.weights[model]
    }

    /// The model to step next among those with work (`ready[m]`), or None.
    pub fn next(&mut self, ready: &[bool]) -> Option<usize> {
        assert_eq!(ready.len(), self.models(), "one entry per model");
        let continuing = (0..self.models())
            .filter(|&model| ready[model] && self.ready[model])
            .map(|model| self.level(model))
            .reduce(f64::min);
        let returning = (0..self.models()).filter(|&model| ready[model] && !self.ready[model]);
        // With no model that kept working, the returning ones start level with each other.
        let reference = continuing.or_else(|| {
            returning
                .clone()
                .map(|model| self.level(model))
                .reduce(f64::max)
        });
        if let Some(reference) = reference {
            for model in returning {
                self.used[model] = self.used[model].max(reference * self.weights[model]);
            }
        }
        self.ready.copy_from_slice(ready);
        (0..self.models())
            .filter(|&model| ready[model])
            .min_by(|&left, &right| self.level(left).total_cmp(&self.level(right)))
    }

    /// Add `seconds` of card time to `model`.
    pub fn charge(&mut self, model: usize, seconds: f64) {
        if seconds.is_finite() && seconds > 0.0 {
            self.used[model] += seconds;
        }
    }

    /// Each model's step seconds per unit of weight.
    pub fn levels(&self) -> Vec<f64> {
        (0..self.models()).map(|model| self.level(model)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Step `rounds` times, each chosen model taking `cost[m]` seconds; returns the time per model.
    fn run(share: &mut ModelShare, ready: &[bool], cost: &[f64], rounds: usize) -> Vec<f64> {
        let mut time = vec![0.0; cost.len()];
        for _ in 0..rounds {
            let model = share.next(ready).unwrap();
            share.charge(model, cost[model]);
            time[model] += cost[model];
        }
        time
    }

    #[test]
    fn busy_models_get_time_in_proportion_to_their_weights() {
        let mut share = ModelShare::new(&[1.0, 3.0]).unwrap();
        let time = run(&mut share, &[true, true], &[0.010, 0.030], 4000);
        let fraction = time[1] / (time[0] + time[1]);
        assert!((fraction - 0.75).abs() < 0.01, "{fraction}");
    }

    #[test]
    fn step_lengths_do_not_change_the_shares() {
        // a long-step model and a short-step model with equal weights each get half the time
        let mut share = ModelShare::new(&[1.0, 1.0]).unwrap();
        let time = run(&mut share, &[true, true], &[0.002, 0.050], 5000);
        let fraction = time[0] / (time[0] + time[1]);
        assert!((fraction - 0.5).abs() < 0.02, "{fraction}");
    }

    #[test]
    fn an_idle_model_banks_no_credit() {
        let mut share = ModelShare::new(&[1.0, 1.0]).unwrap();
        // model 0 works alone for 10 s
        run(&mut share, &[true, false], &[0.01, 0.01], 1000);
        // then model 1 has work: it does not get the next 10 s to itself
        let mut streak = 0;
        while share.next(&[true, true]) == Some(1) {
            share.charge(1, 0.01);
            streak += 1;
            assert!(streak < 3, "model 1 caught up for {streak} steps");
        }
    }

    #[test]
    fn only_models_with_work_run() {
        let mut share = ModelShare::new(&[1.0, 1.0, 1.0]).unwrap();
        assert_eq!(share.next(&[false, false, false]), None);
        for _ in 0..10 {
            let model = share.next(&[false, true, false]).unwrap();
            assert_eq!(model, 1);
            share.charge(model, 1.0);
        }
    }

    #[test]
    fn weights_must_be_positive() {
        assert!(ModelShare::new(&[]).is_err());
        assert!(ModelShare::new(&[1.0, 0.0]).is_err());
        assert!(ModelShare::new(&[f64::NAN]).is_err());
    }
}
