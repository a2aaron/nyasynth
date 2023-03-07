use crate::common::{Note, Vel};

pub struct KeyTracker {
    /// A list of the currently held keys.
    pub held_keys: Vec<(Note, Vel)>,
    /// The note from which the next held note will be pitchbent from. If this is None, then
    /// the next held note will not have any pitchbend.
    pub portamento_key: Option<Note>,
}

impl KeyTracker {
    pub fn new() -> KeyTracker {
        KeyTracker {
            held_keys: Vec::with_capacity(16),
            portamento_key: None,
        }
    }

    /// Handle a NoteOn event. This function returns Some if the note passed into the function should
    /// have portamento, and None if not.
    pub fn note_on(&mut self, note: Note, vel: Vel, polycat: bool) -> Option<Note> {
        self.held_keys.push((note, vel));
        if polycat {
            let portamento = self.portamento_key;
            self.portamento_key = Some(note);
            portamento
        } else {
            match self.held_keys.last().copied() {
                Some((top_note, _)) => Some(top_note),
                None => todo!(),
            }
        }
    }

    /// Handle a NoteOff event. This function returns Some if the note removed would cause the top
    /// of the stack to change. The returned value is the new top of stack. This is used in monocat
    /// mode, where removing the top-most note (aka: the only currently playing note) causes an
    /// internal note on event to occur.
    pub fn note_off(&mut self, note: Note) -> Option<(Note, Vel)> {
        if self.portamento_key == Some(note) {
            self.portamento_key = None;
        }

        // If the released key is actually in the key stack, then remove it. Otherwise, do nothing.
        if let Some(index) = self.held_keys.iter().position(|x| x.0 == note) {
            self.held_keys.remove(index);

            // If the top-of-stack key was released, then we need to return the second to last note
            // if one exists
            let note_on_event = if index == self.held_keys.len() {
                self.held_keys.last().copied()
            } else {
                None
            };

            note_on_event
        } else {
            None
        }
    }
}
