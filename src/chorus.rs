use biquad::{Biquad, ToHertz};

use crate::{
    common::SampleRate,
    params::{ChorusParams, MAX_CHORUS_DEPTH, MAX_CHORUS_DISTANCE},
    sound_gen::{NoteShape, Oscillator},
};

const CHORUS_SIZE: usize = (100.0 + 2.0 * MAX_CHORUS_DEPTH + MAX_CHORUS_DISTANCE) as usize;

pub struct Chorus {
    delay_line: Vec<f32>,
    write_head: usize,
    read_head_oscillator: Oscillator,
    // To remove crackling
    filter: biquad::DirectForm1<f32>,
}

impl Chorus {
    pub fn new(sample_rate: SampleRate) -> Chorus {
        let coefficients = get_coefficients(sample_rate);
        Chorus {
            delay_line: vec![0.0; CHORUS_SIZE],
            write_head: 0,
            read_head_oscillator: Oscillator::new(),
            filter: biquad::DirectForm1::<f32>::new(coefficients),
        }
    }

    pub fn set_sample_rate(&mut self, sample_rate: SampleRate) {
        let new_coefficients = get_coefficients(sample_rate);
        self.filter.update_coefficients(new_coefficients);
    }

    pub fn next_sample(
        &mut self,
        in_sample: f32,
        sample_rate: SampleRate,
        params: &ChorusParams,
        shape: NoteShape,
    ) -> f32 {
        self.write_head = (self.write_head + 1).rem_euclid(self.delay_line.len());
        self.delay_line[self.write_head] = in_sample;

        let read_head_mod =
            self.read_head_oscillator
                .next_sample(sample_rate, shape, params.rate, 0.0);

        let offset = params.min_distance + ((read_head_mod + 1.0) * params.depth);

        let value = self.fractional_lookup(offset);
        self.filter.run(value)
    }

    // Do fractional delay interpolation. The offset value is in samples and will be how many samples
    // behind the write head to look at. This function does linear interpolation.
    fn fractional_lookup(&self, offset: f32) -> f32 {
        let index = self.write_head as f32 - offset;

        let index_upper = index.ceil() as isize;
        let index_lower = index_upper - 1;
        let index_lower2 = index_upper - 2;
        let index_lower3 = index_upper - 3;

        let index_upper = index_upper.rem_euclid(self.delay_line.len() as isize) as usize;
        let index_lower = index_lower.rem_euclid(self.delay_line.len() as isize) as usize;
        let index_lower2 = index_lower2.rem_euclid(self.delay_line.len() as isize) as usize;
        let index_lower3 = index_lower3.rem_euclid(self.delay_line.len() as isize) as usize;

        let index_fractional = index.fract();

        let sample_upper = self.delay_line[index_upper];
        let sample_lower = self.delay_line[index_lower];
        let sample_lower2 = self.delay_line[index_lower2];
        let sample_lower3 = self.delay_line[index_lower3];

        cubic_interpolate(
            sample_lower3,
            sample_lower2,
            sample_lower,
            sample_upper,
            index_fractional,
        )
    }
}

/// Interpolates between x2 and x3, where x0, x1, x2, and x3 are all evenly spaced points.
/// See https://www.desmos.com/calculator/hsiestmj4o for quadratic interpolation.
/// See https://www.desmos.com/calculator/bxcarpst5l for cubic interpolation.
fn cubic_interpolate(x0: f32, x1: f32, x2: f32, x3: f32, t: f32) -> f32 {
    let c0 = (x0 / -6.0) + (x1 / 2.0) + (x2 / -2.0) + (x3 / 6.0);
    let c1 = (x1 / 2.0) - x2 + (x3 / 2.0);
    let c2 = (x0 / 6.0) - x1 + (x2 / 2.0) + (x3 / 3.0);
    let c3 = x2;

    c0 * t * t * t + c1 * t * t + c2 * t + c3
}
fn get_coefficients(sample_rate: SampleRate) -> biquad::Coefficients<f32> {
    biquad::Coefficients::<f32>::from_params(
        biquad::Type::LowPass,
        sample_rate.hz(),
        (sample_rate.get() / 4.0).hz(),
        0.1,
    )
    .unwrap()
}
