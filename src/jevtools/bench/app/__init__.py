"""The app-domain benchmark: realistic assistant tasks over an app's own data (``jevtools bench app``).

Six synthetic domains (email and calendar, CRM, banking, workspace files, IT helpdesk, research data workflows), each
a §11.1 dataset with its tools, its data sources and hand-labelled cases. See ``docs/BENCH.md``.
"""

from jevtools.bench.app.runner import DOMAINS, AppRecord, AppReport, domains_dir, load_domain, run_app, run_domains

__all__ = ["DOMAINS", "AppRecord", "AppReport", "domains_dir", "load_domain", "run_app", "run_domains"]
