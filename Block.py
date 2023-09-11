import numpy as np
from gnuradio import gr, digital

class blk(gr.sync_block):  # other base classes are basic_block, decim_block, interp_block
    """
    Decision Directed PLL - Decision Directed based carrier recovery pll

    Parameters:
        bw - Loop Bandwith (default = 2*pi/100)
        damp - Loop Damping (default = 1.0)
        const - Slicer Constellation (default = BPSK)
    """

    def __init__(self, bw=0.0628, damp=1.0, const=digital.bpsk_constellation()):  # only default arguments here
        """arguments to this function show up as parameters in GRC"""
        gr.sync_block.__init__(
            self,
            name='Decision Directed PLL',   # will show up in GRC
            in_sig=[np.complex64],
            out_sig=[np.complex64, np.float32]
        )
        self.bw = bw
        self.damp = damp

        self.phase = 0
        self.freq = 0
        self.prevgains = [0,0]
        self.slicer_const = const.points()

        self.denom = (len(self.slicer_const) + 2 * self.damp * self.bw + self.bw * self.bw)
        self.alpha = (4 * self.damp * self.bw) /  self.denom
        self.beta = (4 * self.bw * self.bw) / self.denom

    def Decision(self, symbol):
        return self.slicer_const[np.argmin(np.abs(self.slicer_const - symbol))]

    def updateGains(self):
        self.denom = (len(self.slicer_const) + 2 * self.damp * self.bw + self.bw * self.bw) 
        self.alpha = (4 * self.damp * self.bw) /  self.denom
        self.beta = (4 * self.bw * self.bw) / self.denom
        self.phase = 0
        self.freq = 0

    def work(self, input_items, output_items):
        inputs =  input_items[0]
        outputs = output_items[0]
        errors = output_items[1]

        if(self.prevgains[0] != self.bw or self.prevgains[1] != self.damp):
            self.prevgains = [self.bw*1, self.damp*1]
            self.updateGains()

        for i in range(len(inputs)):
            out = inputs[i] * np.exp(-1j*self.phase)
            error = (out * np.conj(self.Decision(out))).imag
            self.freq += self.beta * error
            self.phase += self.freq + self.alpha * error
            self.phase = (np.pi + self.phase) % (2*np.pi) - np.pi
            outputs[i] = out
            errors[i] = error
        return len(outputs)
