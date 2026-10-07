from qsfo.parser import Parser

class OnlineMonitor():
    def __init__(self, 
                 formula: str, 
                 sampling_interval: None | float=None, 
                 horizon: None | float=None,
                 max_samples: None | int=None,
                 csv: None | str=None,
                 csv_with_robusteness: bool=False,
                 csv_no_buffering: bool=False,
                 stdout: bool=True
                 ):
        
        self.formula = Parser(formula)                   # The SFO formula used to calculate the robusteness score
        self.sampling_interval = sampling_interval       # Optional sampling interval (float) for inputs that do not contain the time variable `t` explicitely
        self.horizon = horizon                           # Optional horizon value (float)
        self.max_samples = max_samples                   # Optional maximum number of CSV samples to load from input traces
        self.csv = csv                                   # Write output to the csv file
        self.csv_with_robusteness = csv_with_robusteness # Write robustness signal into CSV (makes sense only if the len of the signal is always 1)
        self.csv_no_buffering = csv_no_buffering         # Write output to CSV immediately, without buffering
        self.stdout = stdout                             # Write not output to stdout

        if stdout:
            print("--- Parsed formula ---")
            print(formula)
            print("------")
            print("Free variables: ", set(map(str, formula.free_variables())) or "∅")
            print("Bound variables: ", set(map(str, formula.bound_variables())) or "∅")
            print("Signals: ", set(map(lambda s: str(s.name()), formula.signals())))


    