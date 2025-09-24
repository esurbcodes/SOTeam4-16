# Team Names:
Wesley Cameron Todd

Esau Cortez

Ethan Surber

Sam Brahim

# Metrics:

# Ramp_up_time: 

Start latency timer using time.perf_counter().

Look for a local README: if resource contains local_dir (a directory path), we check common README filenames (README.md, README.rst, README.txt, README) in that directory. If found, read it (UTF-8, errors replaced).

If no local README, attempt a best-effort remote fetch (only if requests is installed) for common repo hosts:

For GitHub: tries raw.githubusercontent.com/{owner}/{repo}/{branch}/README.md for main and master.

For Hugging Face: tries similar raw/{branch}/README.md patterns.

Generic fallbacks are also attempted.

Note: this remote fetch is optional and will be skipped if requests is not present. (Testing can mock requests so no network is required.)

If no README content is available, return score 0.0 and the elapsed latency.

If README content is found, compute:

Length score from the word count using thresholds (0.0 / 0.1 / 0.25 / 0.4).

Installation score = +0.35 if README contains an "installation" heading or common install phrases (pip install, conda install, docker, etc.).

Code snippet score = +0.25 if README contains fenced code blocks (```) or indented code lines (4 leading spaces or tabs).

Sum weights (length + install + code) and cap at 1.0. Round score to 4 decimals.

Return (score, latency_ms) where latency_ms is integer milliseconds (rounded).
