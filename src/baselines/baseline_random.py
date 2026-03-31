# ==============  SemEval-2015 Task 1  ==============
#  Paraphrase and Semantic Similarity in Twitter
# ===================================================
#
# Author: Wei Xu (UPenn xwe@cis.upenn.edu)
#
# a baseline system that completely use random outputs
#
import logging
import random

from ..paths import SYSTEMOUTPUTS_DIR, TEST_DATA_PATH

logger = logging.getLogger(__name__)

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    outfilename = SYSTEMOUTPUTS_DIR / "PIT2015_BASELINE_01_random_TEST.output"

    ntline = 0
    with TEST_DATA_PATH.open() as tf:
        for tline in tf:
            tline = tline.strip()
            if len(tline.split("\t")) == 7:  # TODO: why changed from 6?
                ntline += 1

    # output the results into a file
    with outfilename.open("w") as outf:
        for _ in range(ntline):
            score = random.random()
            if score >= 0.5:
                outf.write("true\t" + "{0:.4f}".format(score) + "\n")
            else:
                outf.write("false\t" + "{0:.4f}".format(score) + "\n")

    logger.info("Wrote %s predictions to %s", ntline, outfilename)


if __name__ == "__main__":
    main()



