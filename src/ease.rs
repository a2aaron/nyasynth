use std::ops::{Add, Div, Mul, Sub};

pub trait Lerpable = Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Mul<f32, Output = Self>
    + Sized
    + Copy
    + Clone;

pub trait InvLerpable = Sub<Self, Output = Self> + Div<Self, Output = f32> + Sized + Copy + Clone;

/// An enum representing an ease.
pub enum Easing<T> {
    /// Linearly ease from start to end.
    Linear { start: T, end: T },
    /// Linearly ease from start to mid when `t` is between `0.0` and `split_at`,
    /// then linearly ease from mid to end when `t` is between `split_at` and `1.0`
    SplitLinear {
        start: T,
        mid: T,
        end: T,
        split_at: f32,
    },
    /// Linearly ease from start to end, but snap to the given number of steps.
    /// For example, if start = 1.0, end = 2.0, steps = 3, then the valid values
    /// are 1.0, 1.5, and 2.0
    SteppedLinear { start: T, end: T, steps: usize },
    /// Exponentially ease from start to end.
    Exponential { start: T, end: T },
}

impl<T: Lerpable + InvLerpable> Easing<T> {
    /// Ease using the given interpolation value `t`. `t` is expected to be in
    /// [0.0, 1.0] range.
    pub fn ease(&self, t: f32) -> T {
        match *self {
            Easing::Linear { start, end } => lerp(start, end, t),
            Easing::SplitLinear {
                start,
                mid,
                end,
                split_at,
            } => {
                if t < split_at {
                    // Map [0.0, split_at] to the [start, mid] range
                    remap(0.0, split_at, t, start, mid)
                } else {
                    // Map [split_at, 1.0] to the [mid, end] range
                    remap(split_at, 1.0, t, mid, end)
                }
            }
            Easing::SteppedLinear { start, end, steps } => {
                let stepped_t = snap_float(t, steps);
                lerp(start, end, stepped_t)
            }
            Easing::Exponential { start, end } => {
                let expo_t = ease_in_expo(t);
                lerp(start, end, expo_t)
            }
        }
    }

    /// Given a value, return the `t` interpolation value such that `ease(t) == val`.
    /// inv_ease assumes easing functions are invertible, which might not be true
    /// for all functions (ex: SplitLinear that does not ease all the way to 1.0)
    pub fn inv_ease(&self, val: T) -> f32 {
        match *self {
            Easing::Linear { start, end } => inv_lerp(start, end, val),
            Easing::SplitLinear {
                start,
                mid,
                end,
                split_at,
            } => {
                // First determine if the value fits into the lower half of the function
                // Map [start, end] to [0.0, split_at]
                let lower_val = remap(start, mid, val, 0.0, split_at);
                if lower_val < split_at {
                    lower_val
                } else {
                    // Otherwise the value is in the upper half
                    // Map [mid, end] to [split_at, 1.0]
                    remap(mid, end, val, split_at, 1.0)
                }
            }
            Easing::SteppedLinear { start, end, steps } => {
                let t = inv_lerp(start, end, val);
                snap_float(t, steps)
            }
            Easing::Exponential { start, end } => {
                let t = inv_lerp(start, end, val);
                inv_ease_in_expo(t)
            }
        }
    }
}

pub struct DiscreteLinear<T, const N: usize> {
    pub values: [T; N],
}

impl<T: Eq + Copy + Clone, const N: usize> DiscreteLinear<T, N> {
    pub fn ease(&self, t: f32) -> T {
        let index = (t * self.values.len() as f32).floor() as usize;
        self.values[index.clamp(0, self.values.len() - 1)]
    }

    pub fn inv_ease(&self, val: T) -> f32 {
        match self.values.iter().position(|&x| x == val) {
            Some(index) => (index as f32) / (self.values.len() as f32),
            None => 0.0,
        }
    }
}

/// Lerp between two values. This function is clamped.
pub fn lerp<T: Lerpable>(start: T, end: T, t: f32) -> T {
    (end - start) * t.clamp(0.0, 1.0) + start
}

/// Returns the "inverse lerp" of a value. The returned value is zero if val == start
/// and is 1.0 if val == end. This function is clamped to the [0.0, 1.0] range.
pub fn inv_lerp<T: InvLerpable>(start: T, end: T, val: T) -> f32 {
    ((val - start) / (end - start)).clamp(0.0, 1.0)
}

/// Map the range [old_start, old_end] to [new_start, new_end]. Note that
/// lerp(start, end, t) == remap(0.0, 1.0, t, start, end)
/// inv_lerp(start, end, val) == remap(start, end, val, 0.0, 1.0)
pub fn remap<T: InvLerpable, U: Lerpable>(
    old_start: T,
    old_end: T,
    val: T,
    new_start: U,
    new_end: U,
) -> U {
    let t = inv_lerp(old_start, old_end, val);
    lerp(new_start, new_end, t)
}

pub fn ease_in_expo(x: f32) -> f32 {
    if x <= 0.0 {
        0.0
    } else {
        (2.0f32.powf(10.0 * x) - 1.0) / (2.0f32.powf(10.0) - 1.0)
    }
}

pub fn inv_ease_in_expo(x: f32) -> f32 {
    if x <= 0.0 {
        0.0
    } else {
        ((2.0f32.powf(10.0) - 1.0) * x + 1.0).log2() / 10.0
    }
}

pub fn ease_in_poly(x: f32, i: i32) -> f32 {
    x.powi(i)
}

/// Snap a float value in range 0.0-1.0 to the nearest f32 region
/// For example, snap_float(_, 4) will snap a float to either:
/// 0.0, 0.333, 0.666, or 1.0
pub fn snap_float(value: f32, num_regions: usize) -> f32 {
    // We subtract one from this denominator because we want there to only be
    // four jumps. See also https://www.desmos.com/calculator/esnnnbfzml
    let num_regions = num_regions as f32;
    (num_regions * value).floor() / (num_regions - 1.0)
}
