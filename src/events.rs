use std::time::{Duration, Instant};

pub enum Action {
    None,
    ShowOverlay,
    HideOverlay,
}

enum InternalState {
    Activated(Instant),
    Refractory,
    Armed,
}

pub struct State {
    visible: bool,
    state: InternalState,
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

    pub fn new(delay: Duration) -> Self {
        Self {
            visible: true,
            state: InternalState::Armed,
            delay,
        }
    }
    pub(crate) fn handle<Vr: crate::vrapi::Vr>(&mut self, vrsys: &Vr) -> Result<(), Vr::Error> {
        let mut button_pressed = 0;
        if vrsys.get_action_state(crate::vrapi::Action::Button1)? {
            button_pressed += 1;
        }
        if vrsys.get_action_state(crate::vrapi::Action::Button2)? {
            button_pressed += 1;
        }
        match (&self.state, button_pressed) {
            (InternalState::Refractory, 0) => {
                self.state = InternalState::Armed;
            }
            (InternalState::Refractory, _) => (),
            (InternalState::Activated(_), 0) | (InternalState::Activated(_), 1) => {
                self.state = InternalState::Armed;
            }
            (InternalState::Activated(_), _) => (),
            (InternalState::Armed, 2) => {
                self.state = InternalState::Activated(Instant::now());
            }
            (InternalState::Armed, _) => (),
        }
        Ok(())
    }
    pub fn turn(&mut self) -> Action {
        if let InternalState::Activated(start) = self.state {
            if !self.visible && std::time::Instant::now() - start > self.delay {
                self.state = InternalState::Refractory;
                self.visible = true;
                Action::ShowOverlay
            } else if self.visible {
                self.state = InternalState::Refractory;
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
