from mrjob.job import MRJob
from mrjob.step import MRStep
import re

class LongWrds(MRJob):

    # Mapper
    def map_fn(self, _, line):
        pat = r'\b[a-zA-Z]+\b'
        wrds = re.findall(pat, line)

        for w in wrds:
            init_c = w.lower()[0]
            yield init_c, w.lower()

    # Combiner
    def cmb_fn(self, init_c, wrds):
        yield init_c, list(self.long_w(wrds, single_inp=True))

    # Reducer
    def red_fn(self, init_c, wrds):
        yield init_c, list(self.long_w(wrds, single_inp=False))

    # Helper
    def long_w(self, wrds, single_inp):
        lng_w = set()
        max_ln = 0

        if not single_inp:
            wrds = [w for sublist in wrds for w in sublist]

        for w in wrds:
            if len(w) > max_ln:
                max_ln = len(w)
                lng_w = {w}
            elif len(w) == max_ln:
                lng_w.add(w)
        return lng_w

    # Steps
    def steps(self):
        return [
            MRStep(mapper=self.map_fn, combiner=self.cmb_fn, reducer=self.red_fn)
        ]

if __name__ == '__main__':
    LongWrds.run()
