Requirements

Python 3.9 or newer
Libraries: pandas, numpy, scipy, scikit-learn 1.2 or newer, and matplotlib

Install with:

pip install pandas numpy scipy scikit-learn matplotlib

How to run
Put the CSV in the same folder as the scripts, then run python workflow.py, python report_numbers.py or python figures.py. To skip the 2-minute evaluation, set RUN_EXPERIMENT = False in report_numbers.py.

I used 1.2 as the minimum for scikit-learn because figures.py uses an option (sparse_output) that older versions don't have. If you want, I can add a line for table6_results.csv as an output file.
