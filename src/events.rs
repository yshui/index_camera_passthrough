use std::time::{Duration, Instant};

use openvr_sys2::EVREventType;

pub enum Action {
    None,
    ShowOverlay,
    HideOverlay,
}

enum InternalState {
    NoButtonPressed,
    OneButtonPressed,
    TwoButtonPressed(Instant),
    PostTransition,
}

pub struct State {
    visible: bool,
    state: InternalState,
    button: crate::config::Button,
    delay: Duration,
}

impl State {
    /// Whether the overlay should be visible
    pub fn visible(&self) -> bool {
        self.visible
    }

    /// How long should we wait between frames
    pub fn interval(&self) -> Duration {
        if self.visible {
            Duration::ZERO
        } else {
            Duration::from_millis(100)
        }
    }

    pub fn new(button: crate::config::Button, delay: Duration) -> Self {
        Self {
            visible: true,
            button,
            state: InternalState::NoButtonPressed,
            delay,
        }
    }
    pub fn handle(&mut self, event: &openvr_sys2::VREvent_t) {
        if event.eventType == EVREventType::VREvent_ButtonPress as u32 {
            let button = unsafe { event.data.controller.button };
            if button == Into::<openvr_sys2::EVRButtonId>::into(self.button) as u32 {
                match self.state {
                    InternalState::NoButtonPressed => {
                        self.state = InternalState::OneButtonPressed;
                    }
                    InternalState::OneButtonPressed => {
                        self.state = InternalState::TwoButtonPressed(Instant::now());
                    }
                    InternalState::TwoButtonPressed(_) | InternalState::PostTransition => (),
                }
            }
        } else if event.eventType == EVREventType::VREvent_ButtonUnpress as u32 {
            let button = unsafe { event.data.controller.button };
            if button == Into::<openvr_sys2::EVRButtonId>::into(self.button) as u32 {
                match self.state {
                    // Released a button when no button is pressed?
                    InternalState::NoButtonPressed => (),
                    InternalState::OneButtonPressed => {
                        self.state = InternalState::NoButtonPressed;
                    }
                    InternalState::TwoButtonPressed(_) | InternalState::PostTransition => {
                        self.state = InternalState::OneButtonPressed;
                    }
                }
            }
        }
    }
    pub fn turn(&mut self) -> Action {
        if let InternalState::TwoButtonPressed(start) = self.state {
            if !self.visible && std::time::Instant::now() - start > self.delay {
                self.state = InternalState::PostTransition;
                self.visible = true;
                Action::ShowOverlay
            } else if self.visible {
                self.state = InternalState::PostTransition;
                self.visible = false;
                Action::HideOverlay
            } else {
                Action::None
            }
        } else {
            Action::None
        }
    }
}
