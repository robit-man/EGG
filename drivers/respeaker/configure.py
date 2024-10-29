import curses
import struct
import usb.core
import usb.util
import time
from tuning import PARAMETERS, Tuning

# Insert your PARAMETERS dictionary and Tuning class definitions here

class InteractiveTuningUI:
    def __init__(self, tuning):
        self.tuning = tuning
        self.selected_param = 0
        self.params_list = list(PARAMETERS.keys())

    def get_param_info(self, param_name):
        return PARAMETERS.get(param_name, [])

    def display_params(self, stdscr):
        stdscr.clear()
        stdscr.addstr(0, 0, "Interactive Microphone Control System", curses.A_BOLD | curses.A_UNDERLINE)
        stdscr.addstr(2, 0, "Navigate with UP/DOWN arrows. Press Enter to read/write a parameter.")
        stdscr.addstr(3, 0, "Press 'q' to quit.")

        for idx, param_name in enumerate(self.params_list):
            info = self.get_param_info(param_name)
            is_rw = 'rw' if info[5] == 'rw' else 'ro'
            val = self.tuning.read(param_name) if info else "N/A"

            line = f"{param_name:20} {val:15} {is_rw:2} {info[6]:<50}"
            if idx == self.selected_param:
                stdscr.addstr(idx + 5, 0, line, curses.A_REVERSE)
            else:
                stdscr.addstr(idx + 5, 0, line)
        stdscr.refresh()

    def handle_user_input(self, stdscr):
        key = stdscr.getch()
        
        # Navigate the parameter list
        if key == curses.KEY_UP and self.selected_param > 0:
            self.selected_param -= 1
        elif key == curses.KEY_DOWN and self.selected_param < len(self.params_list) - 1:
            self.selected_param += 1
        elif key == ord('q'):
            return False
        elif key == ord('\n'):
            self.edit_param(stdscr)
        
        return True

    def edit_param(self, stdscr):
        param_name = self.params_list[self.selected_param]
        param_info = self.get_param_info(param_name)

        if param_info[5] == 'rw':  # Check if the parameter is writable
            min_val, max_val = param_info[4], param_info[3]
            current_val = self.tuning.read(param_name)
            prompt = f"Enter new value for {param_name} (Current: {current_val}, Min: {min_val}, Max: {max_val}): "
            
            # Display prompt at the bottom
            stdscr.addstr(curses.LINES - 2, 0, prompt)
            stdscr.clrtoeol()
            stdscr.refresh()

            # Capture input at bottom and handle escape to cancel
            curses.echo()
            input_win = curses.newwin(1, curses.COLS, curses.LINES - 1, 0)
            input_win.clear()
            input_win.refresh()

            new_val = input_win.getstr().decode("utf-8").strip()
            curses.noecho()

            # Check for Escape (if input is empty, treat as Escape)
            if new_val == "":
                stdscr.addstr(curses.LINES - 1, 0, "Input canceled. Press any key to continue.")
                stdscr.getch()
                return

            # Validate and apply value if within bounds
            try:
                if param_info[2] == 'int':
                    new_val = int(new_val)
                elif param_info[2] == 'float':
                    new_val = float(new_val)

                # Check range constraints
                if new_val < min_val or new_val > max_val:
                    raise ValueError("Value out of range")
                
                # Apply the new value
                self.tuning.write(param_name, new_val)
                stdscr.addstr(curses.LINES - 1, 0, f"Set {param_name} to {new_val}. Press any key to continue.")
            except ValueError as e:
                stdscr.addstr(curses.LINES - 1, 0, f"Invalid input: {e}. Press any key to continue.")
            stdscr.getch()

    def run(self, stdscr):
        curses.curs_set(0)  # Hide cursor
        stdscr.nodelay(1)   # Make getch non-blocking

        while True:
            self.display_params(stdscr)
            if not self.handle_user_input(stdscr):
                break
            time.sleep(0.1)  # Control the refresh rate for parameter updates

def main(stdscr):
    # Set up the USB device and Tuning interface
    dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    if dev is None:
        print("USB device not found.")
        return
    mic_tuning = Tuning(dev)
    
    ui = InteractiveTuningUI(mic_tuning)
    ui.run(stdscr)

# Start curses
curses.wrapper(main)
