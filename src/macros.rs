#![macro_use]

#[macro_export]
macro_rules! generate_raw_params {
   ($raw_parameters: ident, $parameter_type: ident;
    $($variant:pat, $field_name:ident, $name:expr, $idx:expr, $default:expr, $easer_field:ident, $preset_field:expr;)*) => {
        /// The raw parameter values that a host DAW will set and modify.
        /// These are unscaled and are always in the [0.0, 1.0] range
        pub struct $raw_parameters {
            $(pub $field_name: AtomicFloat,)*
            /// The host callback, used for communicating with the VST host
            pub host: vst::plugin::HostCallback,
            /// The sender that notifies the GUI thread to update due to the host
            /// modifying a value. This is unwind safe because if opening a window
            /// panics for some reason, we aren't going to use the sender anyways
            pub sender: std::panic::AssertUnwindSafe<tokio::sync::broadcast::Sender<(ParameterType, f32)>>,
        }
    };
}

#[macro_export]
macro_rules! impl_new {
    ($raw_parameters: ident, $parameter_type: ident;
     $($variant:pat, $field_name:ident, $name:expr, $idx:expr, $default:expr, $easer_field:ident, $preset_field:expr;)*) => {
        impl $raw_parameters {
            pub fn new(host: vst::plugin::HostCallback) -> Self {
                $raw_parameters {
                    $($field_name: vst::util::AtomicFloat::new($default),)*
                    host,
                    sender: std::panic::AssertUnwindSafe(tokio::sync::broadcast::channel(128).0), // TODO: what size of channel should this be?
                }
            }
        }
    };
}

#[macro_export]
macro_rules! impl_get_ref {
    ($raw_parameters: ident, $parameter_type: ident;
     $($variant:pat, $field_name:ident, $name:expr, $idx:expr, $default:expr, $easer_field:ident, $preset_field:expr;)*) => {
        impl $raw_parameters {
            pub fn get_ref(&self, x: $parameter_type) -> &vst::util::AtomicFloat {
                match x {
                    $($variant => &self.$field_name,)*
                }
            }
        }
    };
}

#[macro_export]
macro_rules! impl_get_default {
   ($raw_parameters: ident, $parameter_type: ident;
    $($variant:pat, $field_name:expr, $name:expr, $idx:expr, $default:expr, $easer_field:ident, $preset_field:expr;)*) => {
        impl $raw_parameters {
            pub fn get_default(x: $parameter_type) -> f32 {
                match x {
                    $($variant => $default,)*
                }
            }
        }
    };
}

#[macro_export]
macro_rules! impl_display {
   ($raw_parameters: ident, $parameter_type: ident;
    $($variant:pat, $field_name:expr, $name:expr, $idx:expr, $default:expr, $easer_field:ident, $preset_field:expr;)*) => {
        impl std::fmt::Display for $parameter_type {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self {
                    $($variant => write!(f, $name),)*
                }
            }
        }
    };
}

#[macro_export]
macro_rules! impl_from_i32 {
   ($raw_parameters: ident, $parameter_type: ident;
    $($variant:expr, $field_name:expr, $name:expr, $idx:expr, $default:expr, $easer_field:ident, $preset_field:expr;)*) => {
        impl std::convert::TryFrom<i32> for $parameter_type {
            type Error = ();
            fn try_from(x: i32) -> Result<Self, Self::Error> {
                // Dummy values for parameter indicies
                if x <= -1 {
                    return Err(())
                }
                match x {
                    $($idx => Ok($variant),)*
                    _ => Err(()),
                }
            }
        }
    }
}

#[macro_export]
macro_rules! impl_into_i32 {
   ($raw_parameters: ident, $parameter_type: ident;
    $($variant:pat, $field_name:expr, $name:expr, $idx:expr, $default:expr, $easer_field:ident, $preset_field:expr;)*) => {
        impl From<$parameter_type> for i32 {
            fn from(x: $parameter_type) -> i32 {
                match x {
                    $($variant => $idx,)*
                }
            }
        }
    };
}

macro_rules! set_by_preset_helper {
    ($self:ident, $variant:expr, NONE,               $preset:ident, $($preset_field:ident)+) => {};
    ($self:ident, $variant:expr, $easer_field:ident, $preset:ident, NONE) => {};
    ($self:ident, $variant:expr, $easer_field:ident, $preset:ident, $($preset_field:ident)+) => {
        $self.set_and_update_knob(EASER.$easer_field.inv_ease($preset.$($preset_field).+), $variant);
    };
}

#[macro_export]
macro_rules! impl_set_by_preset {
   ($raw_parameters: ident, $parameter_type: ident;
    $($variant:expr, $field_name:expr, $name:expr, $idx:expr, $default:expr, $easer_field:ident, $($preset_field:ident).+;)*) => {
        impl $raw_parameters {
            /// Set all the parameters via a preset. Note that this also updates the GUI
            pub fn set_by_preset(&self, preset: &PresetData) {
                $(set_by_preset_helper! {self, $variant, $easer_field, preset, $($preset_field)+})*
            }
        }
    }
}
