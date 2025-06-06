import datetime
import time


class PrecisionTimer:
    def __init__(self):
        """
        Creates a precision timer to time the training.
        Starts the timer when it is created.
        """

        self.timer_name = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.start_time = time.perf_counter() # Actually "starts the timer"
        self.is_running = True # Is the timer running
        self.additional_time = 0 # Additional time to add to the elapsed time
        self.end_time = None
        self.precise_elapsed_time = None


    def set_additional_time(self, additional_time):
        self.additional_time = additional_time


    def stop_timer(self):
        """
        Stops the timer.
        """
        
        self.end_time = time.perf_counter()


    def get_elapsed_time(self):
        """
        Stops the timer if it's running,
        then returns the precise elapsed time in seconds.
        
        Returns:
            precise_time (float): The precise elapsed time in seconds.
        """
        self.stop_timer() # generates new end_time
        self.precise_elapsed_time = self.end_time - self.start_time + self.additional_time # Calculates the elapsed time
        
        return self.precise_elapsed_time
        
        
    def get_timer_name(self):
        """
        Returns the timer's name.
            Format: "Year-month-day Hours:Minutes:Seconds"
        
        Returns:
            timer_name (str): The timer name
        """
        
        return self.timer_name
    

    def is_timer_running(self):
        """
        Says if the timer si running or not.
        
        Returns :
            is_running (bool): True is the timer is running
        """
        
        return self.is_running
