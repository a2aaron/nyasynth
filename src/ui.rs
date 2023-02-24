use std::sync::Arc;
use std::{ffi::c_void, hash::Hash};

use iced::{button, futures, Column, Command, Subscription};

use iced_baseview::{Application, WindowSubs};

use raw_window_handle::{HasRawWindowHandle, RawWindowHandle};
use vst::{editor::Editor, host::Host};

use crate::{
    params::{
        ModBankParameter, ModBankType, ModulationSend, ModulationType, OSCType, ParameterType,
        Parameters, RawParameters,
    },
    presets::{try_parse_file, PresetData},
    ui_tabs::{MainTab, ModulationTab, PresetTab, Tabs},
};

const WINDOW_WIDTH: usize = 800;
const WINDOW_HEIGHT: usize = 650;

/// A GUI message.
#[derive(Debug, Clone)]
pub enum Message {
    /// This indicates that a modulation bank has had its send parameter changed.
    ModBankSendChanged(ModBankType, ModulationSend),
    /// This indicates that the GUI had changed the ModulationType parameter
    OSC2ModChanged(ModulationType),
    /// These indicate that the GUI has changed a parameter via a knob
    ParameterChanged(ParameterType, f32),
    /// These are called to make the GUI update itself. Usually, this means that
    /// the host has changed a parameter, so the GUI should change to match.
    /// Note that this works because sending a message also causes `iced` to
    /// call `view` and do a screen update. If it didn't do that, then this won't
    /// work.
    ForceRedraw(ParameterType, f32),
    ChangeTab(Tabs),
    /// Indicates that a new preset has been loaded.
    LoadPreset(std::path::PathBuf),
    /// Save the current parameters as preset at the given folder path. The name of the
    /// preset is the String.
    SaveParamsAsPreset(std::path::PathBuf),
}

/// The struct which manages the GUI. This struct handles both the messages
/// passed to it by iced as well as communciating with host VST for GUI stuff.
pub struct UIFrontEnd {
    main_tab: MainTab,
    modulation_tab: ModulationTab,
    preset_tab: PresetTab,
    selected_tab: Tabs,
    main_button_state: button::State,
    modulation_button_state: button::State,
    preset_button_state: button::State,
    sender: Option<iced_baseview::WindowHandle<Message>>,
    // This is used so that the GUI can update the shared parameters object when
    // a GUI element is changed.
    params: std::sync::Arc<RawParameters>,
    // This just tracks if the control key is held down. This is needed because
    // the VST host captures keyboard events, so we we need to manually track
    // keydown/keyup events.
    control_key: keyboard_types::KeyState,
}

impl Application for UIFrontEnd {
    type Message = Message;
    type Executor = iced_baseview::executor::Default;
    type Flags = Arc<RawParameters>;

    fn new(flag: Self::Flags) -> (Self, Command<Self::Message>) {
        let params = flag;
        let ui = UIFrontEnd {
            main_tab: MainTab::new(params.as_ref()),
            preset_tab: PresetTab::new(params.as_ref()),
            modulation_tab: ModulationTab::new(params.as_ref()),
            sender: None,
            selected_tab: Tabs::Main,
            main_button_state: button::State::new(),
            modulation_button_state: button::State::new(),
            preset_button_state: button::State::new(),
            params,
            control_key: keyboard_types::KeyState::Up,
        };
        (ui, Command::none())
    }

    /// React to an incoming message
    fn update(&mut self, message: Self::Message) -> Command<Self::Message> {
        // TODO: look into moving message handling code out of the update functions
        // for the tabs. It seems like it's bizarrely messy to do it there.
        // Tabs don't really need to know about what messages have happened anyways.
        match message.clone() {
            // The GUI has changed a parameter via knob
            Message::ParameterChanged(param, value) => {
                // We set the parameter according to the changed value.
                self.params.set(value, param);

                // If the knob changed was a "snapping" knob, make sure the knob
                // ends up appearing snapped.
                match param {
                    ParameterType::OSC1(osc_param) => {
                        self.main_tab
                            .update_snapping_knobs(OSCType::OSC1, osc_param);
                    }
                    ParameterType::OSC2(osc_param) => {
                        self.main_tab
                            .update_snapping_knobs(OSCType::OSC2, osc_param);
                    }
                    // This is definitely unreachable because OSC2Mod handling is
                    // done by the OSC2ModChanged message
                    ParameterType::OSC2Mod => unreachable!(),
                    // Also unreachable, this is handled by ModBankSendChanged
                    ParameterType::ModBankSend(_) => unreachable!(),
                    // Don't do anything special for these parameter types.
                    ParameterType::MasterVolume => (),
                    ParameterType::ModBank(_) => {
                        // TODO!
                    }
                }
            }
            // The GUI has changed the modulation type
            Message::OSC2ModChanged(mod_type) => {
                // TODO: Consider using an actual parameter instead of RawParameters
                self.params.set(mod_type.into(), ParameterType::OSC2Mod);
            }
            Message::ModBankSendChanged(mod_bank, mod_send) => {
                self.params
                    .set(mod_send.into(), ParameterType::ModBankSend(mod_bank));
            }
            // The host has changed a parameter, or a redraw was requested
            // We update the knobs based on the current parameter values
            Message::ForceRedraw(param, value) => match param {
                ParameterType::MasterVolume => self.main_tab.master_vol.set_normal(value.into()),
                ParameterType::OSC2Mod => (),
                ParameterType::OSC1(param) => self.main_tab.osc_1.set_knob(param, value),
                ParameterType::OSC2(param) => self.main_tab.osc_2.set_knob(param, value),
                ParameterType::ModBankSend(_) => (),
                ParameterType::ModBank(param) => match param {
                    ModBankParameter::Env1(param) => {
                        self.modulation_tab.env_1.set_knob(param, value)
                    }
                    ModBankParameter::Env2(param) => {
                        self.modulation_tab.env_2.set_knob(param, value)
                    }
                },
            },
            Message::ChangeTab(tab) => {
                self.selected_tab = tab;
                if tab == Tabs::Preset {
                    log::info!("Loading presets from {}", crate::DATA_DIR.display());
                    self.preset_tab.get_presets_from_folder(&*crate::DATA_DIR);
                }
            }
            Message::LoadPreset(path) => match try_parse_file(&path) {
                Ok(preset) => {
                    self.params.set_by_preset(&preset);
                    log::info!("Loaded {}!", path.display());
                }
                Err(err) => log::info!("Couldn't parse {}: {:?}", path.display(), err),
            },
            Message::SaveParamsAsPreset(folder) => {
                let (path, i) = crate::presets::get_free_file_name(folder, "preset", "json");
                let name = format!("Unnamed Preset {}", i);
                match crate::presets::save_preset_to_file(
                    PresetData::from_params(&Parameters::from(self.params.as_ref()), name),
                    &path,
                ) {
                    Ok(()) => log::info!("Saved preset to {}", path.display()),
                    Err(err) => log::info!("Couldn't save preset to {}: {:?}", path.display(), err),
                }
                self.preset_tab.get_presets_from_folder(&*crate::DATA_DIR);
            }
        }

        // Make the VST host update its own parameter display. This is needed
        // so the host actually updates with GUI.
        match message {
            Message::OSC2ModChanged(_)
            | Message::ParameterChanged(_, _)
            | Message::ModBankSendChanged(_, _)
            | Message::LoadPreset(_) => self.params.host.update_display(),
            _ => (),
        }
        Command::none()
    }

    /// Set up the GUI elements
    /// Note that this isn't called every frame--it's only called when there's
    /// an update to the view (which happens to be only when messages happen)
    fn view(&mut self) -> iced::Element<'_, Self::Message> {
        let (screen_width, screen_height) = (self.size().0 as u32, self.size().1 as u32);
        let params = self.params.as_ref();
        let ui_body = match self.selected_tab {
            Tabs::Main => self.main_tab.view(screen_width, screen_height, params),
            Tabs::Preset => self.preset_tab.view(screen_width, screen_height, params),
            Tabs::Modulation => self
                .modulation_tab
                .view(screen_width, screen_height, params),
        };

        let main_tab = make_button(
            &mut self.main_button_state,
            "Main",
            Message::ChangeTab(Tabs::Main),
        );
        let preset_tab = make_button(
            &mut self.preset_button_state,
            "Preset",
            Message::ChangeTab(Tabs::Preset),
        );
        let modulation_tab = make_button(
            &mut self.modulation_button_state,
            "Modulation",
            Message::ChangeTab(Tabs::Modulation),
        );
        let tab_buttons = iced::Row::with_children(vec![
            main_tab.into(),
            modulation_tab.into(),
            preset_tab.into(),
        ]);
        Column::with_children(vec![tab_buttons.into(), ui_body]).into()
    }

    /// This allows iced to recieve messages from the notifier
    fn subscription(
        &self,
        _window_subs: &mut WindowSubs<Self::Message>,
    ) -> Subscription<Self::Message> {
        let recipe = ReceiverRecipe {
            receiver: self.params.sender.subscribe(),
        };
        Subscription::from_recipe(recipe)
    }

    fn renderer_settings() -> iced_baseview::renderer::Settings {
        // We get the settings from the environment so that we can set the
        // renderer backend manually. This fixes a crash on Ableton when the
        // backend is DX12 instead of DX11 or Vulkan.
        log::info!(
            "Getting renderer settings via env vars! WGPU_BACKEND={:?}",
            std::env::var("WGPU_BACKEND"),
        );
        iced_baseview::renderer::Settings::from_env()
    }
}

pub fn make_button<'a>(
    state: &'a mut button::State,
    label: &str,
    msg: Message,
) -> iced::Button<'a, Message> {
    iced::Button::new(state, iced::Text::new(label)).on_press(msg)
}

impl Editor for UIFrontEnd {
    fn size(&self) -> (i32, i32) {
        (WINDOW_WIDTH as i32, WINDOW_HEIGHT as i32)
    }

    fn position(&self) -> (i32, i32) {
        (0, 0)
    }

    fn open(&mut self, parent: *mut c_void) -> bool {
        #[cfg(target_os = "windows")]
        let parent = to_windows_handle(parent);

        #[cfg(target_os = "macos")]
        let parent = to_macos_handle(parent);

        #[cfg(not(any(target_os = "macos", target_os = "windows")))]
        error!("currently unsupported os!");

        let settings = iced_baseview::Settings {
            window: baseview::WindowOpenOptions {
                title: "Revisit".to_string(),
                size: baseview::Size::new(self.size().0 as f64, self.size().1 as f64),
                scale: baseview::WindowScalePolicy::SystemScaleFactor,
            },
            flags: self.params.clone(),
        };

        // For some reason, Ableton will crash upon loading the GUI for our VST
        // if the WGPU backend is DX12 but not Vulkan or DX11. Even weirder, this
        // crash does not occur in any other host I tested (VSTHost, LMMS) reguardless
        // of backend. The crash also didn't occur when running Ableton thru
        // RenderDoc.
        // As a workaround, we will always force Vulkan as the backend, as that
        // seems to not crash.
        // TODO: Try cycling thru backends on crashes?
        // Check here for backends: https://docs.rs/iced_wgpu/0.4.0/iced_wgpu/settings/struct.Settings.html
        #[cfg(target_os = "windows")]
        std::env::set_var("WGPU_BACKEND", "vulkan");

        // TODO: For some reason, this function will panic when running the VST
        // on Ableton. I have no idea why this is the case, but it has something
        // to do with wgpu being unable to create a graphics pipeline. In order
        // to prevent Ableton from choking, I instead catch the panic here and
        // say that opening the window failed.
        let sender = std::panic::catch_unwind(|| {
            iced_baseview::IcedWindow::<UIFrontEnd>::open_parented(&parent, settings)
        });

        match sender {
            Ok(sender) => {
                self.sender = Some(sender);
                true
            }
            Err(err) => {
                log::info!("Couldn't open window! {:#?}", err);
                self.sender = None;
                false
            }
        }
    }

    fn idle(&mut self) {}

    fn close(&mut self) {
        self.sender = None;
    }

    fn is_open(&mut self) -> bool {
        self.sender.is_some()
    }

    fn key_up(&mut self, keycode: vst::editor::KeyCode) -> bool {
        if let Some(sender) = &mut self.sender {
            match keycode.key {
                vst::editor::Key::Control | vst::editor::Key::Shift => {
                    if self.control_key == keyboard_types::KeyState::Down {
                        let event = to_keyboard_event(keycode, keyboard_types::KeyState::Up);
                        sender
                            .send_baseview_event(baseview::Event::Keyboard(event))
                            .expect("Couldn't send message!");
                        self.control_key = keyboard_types::KeyState::Up;
                        return true;
                    }
                }
                _ => (),
            }
        }
        false
    }

    fn key_down(&mut self, keycode: vst::editor::KeyCode) -> bool {
        if let Some(sender) = &mut self.sender {
            match keycode.key {
                vst::editor::Key::Control | vst::editor::Key::Shift => {
                    if self.control_key == keyboard_types::KeyState::Up {
                        let event = to_keyboard_event(keycode, keyboard_types::KeyState::Down);
                        sender
                            .send_baseview_event(baseview::Event::Keyboard(event))
                            .expect("Couldn't send message!");
                        self.control_key = keyboard_types::KeyState::Down;
                        return true;
                    }
                }
                _ => (),
            }
        }
        false
    }
}

/// Convert a `vst::Keycode` + `keyboard_types::KeyState` into an actual
/// `keyboard_types::KeyboardEvent`. This is minimal and pretty much only works
/// with the Control key, so you'll need to fix it if you wanna do more with it.
/// This is used to send keyboard events into `iced_baseview`.
fn to_keyboard_event(
    _: vst::editor::KeyCode,
    state: keyboard_types::KeyState,
) -> keyboard_types::KeyboardEvent {
    match state {
        keyboard_types::KeyState::Down => keyboard_types::KeyboardEvent {
            state,
            key: keyboard_types::Key::Control,
            code: keyboard_types::Code::ControlLeft,
            location: keyboard_types::Location::Standard,
            modifiers: keyboard_types::Modifiers::CONTROL,
            repeat: false,
            is_composing: false,
        },
        keyboard_types::KeyState::Up => keyboard_types::KeyboardEvent {
            state,
            key: keyboard_types::Key::Control,
            code: keyboard_types::Code::ControlLeft,
            location: keyboard_types::Location::Standard,
            modifiers: keyboard_types::Modifiers::empty(),
            repeat: false,
            is_composing: false,
        },
    }
}

struct WindowHandle(RawWindowHandle);

unsafe impl HasRawWindowHandle for WindowHandle {
    fn raw_window_handle(&self) -> RawWindowHandle {
        self.0
    }
}

#[cfg(target_os = "windows")]
fn to_windows_handle(parent: *mut c_void) -> WindowHandle {
    use raw_window_handle::windows::WindowsHandle;
    let mut handle = WindowsHandle::empty();
    handle.hwnd = parent;
    handle.hinstance = std::ptr::null_mut();
    WindowHandle(RawWindowHandle::Windows(handle))
}

#[cfg(target_os = "macos")]
fn to_macos_handle(parent: *mut c_void) -> WindowHandle {
    use raw_window_handle::macos::MacOSHandle;
    let mut handle = MacOSHandle::empty();
    handle.ns_view = parent;
    handle.ns_window = std::ptr::null_mut();
    WindowHandle(RawWindowHandle::MacOS(handle))
}

/// A struct implementing `iced_native::Recipe`. This will send a
/// `Message::ForceRedraw` whenever the `notifier` is recieves a notification.
/// This struct is used to make the iced GUI update whenever the VST host alters
/// a parameter (say, via an automation curve).
struct ReceiverRecipe {
    receiver: tokio::sync::broadcast::Receiver<(ParameterType, f32)>,
}

impl<H, I> iced_native::subscription::Recipe<H, I> for ReceiverRecipe
where
    H: std::hash::Hasher,
{
    type Output = Message;

    fn hash(&self, state: &mut H) {
        // generic hash implementation. this isn't actually very important to us
        std::any::TypeId::of::<Self>().hash(state);
    }

    // This stream function was a nightmare to figure out initally, but here's
    // how it works (thank you to Cassie for explaining this to me!)
    // The function returns a `futures::BoxStream`, which is a async thing.
    // The BoxStream has a function that returns an `Option((message, next_state))`
    // If it returns None, then the stream ends (we don't want this)
    // If it returns `Some((message, next_state))` then we fire off the `message`
    // to iced, and the `next_state` becomes the argument for the input of the
    // function the next time it is run (which is... immediately, i think?)
    // Hence we essentially just run in a look and yield every so often.
    fn stream(
        self: Box<Self>,
        _: futures::stream::BoxStream<'static, I>,
    ) -> futures::stream::BoxStream<'static, Self::Output> {
        Box::pin(futures::stream::unfold(
            self.receiver,
            |mut receiver| async move {
                // Wait for the receiver to recieve a notification before firing
                // the message.
                let result = receiver.recv().await;
                match result {
                    Ok((param, value)) => Some((Message::ForceRedraw(param, value), receiver)),
                    Err(_) => panic!("Reciever couldn't recieve"),
                }
            },
        ))
    }
}
