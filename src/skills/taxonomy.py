"""Curated taxonomy of data-job skills.

Each ``Skill`` has a canonical ``name``, a ``category``, and either:
  * ``aliases`` — literal surface forms matched with word boundaries (the common
    case; works because names like "PostgreSQL" or "Tableau" are unambiguous), or
  * ``pattern`` — an explicit regex for *ambiguous* names that need context
    (``R`` vs "R&D", ``Go`` vs the verb, ``Excel`` vs "excel" the verb).

Extending this is the main way to grow the analysis — add an entry and it shows
up everywhere downstream.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Skill:
    name: str
    category: str
    aliases: tuple[str, ...] = ()
    pattern: str | None = None  # explicit regex; overrides aliases when set


# --- Shared regex fragments for the ambiguous single-token languages ---------
# Variable-width look-behind is unsupported by `re`, so these patterns match the
# disambiguating context *together with* the token (we only need presence).
_SEP = r"(?:\s*[/,]\s*|\s+(?:and|or)\s+)"
_R_NEIGHBORS = r"python|sql|sas|stata|matlab|java|scala|spss|tableau"
_GO_NEIGHBORS = r"python|java|rust|scala|kotlin|ruby"

_R_PATTERN = (
    r"(?i:\bR\s*studio\b)"
    r"|(?i:\bR\s+(?:programming|language|shiny|markdown|packages?|scripts?)\b)"
    r"|(?i:\b(?:" + _R_NEIGHBORS + r")" + _SEP + r")R\b(?![\w&/])"
    r"|\bR" + _SEP + r"(?i:" + _R_NEIGHBORS + r")\b"
    r"|(?i:\b(?:using|with|in|learn|knowledge of|proficient in|"
    r"languages? such as|languages? like|languages? including)\s+)R\b(?![\w&/])"
)

_GO_PATTERN = (
    r"\bGolang\b"
    r"|(?i:\bGo\s+(?:programming|language|developer|lang)\b)"
    r"|(?i:\b(?:" + _GO_NEIGHBORS + r")" + _SEP + r")Go\b(?![\w\-])"
    r"|\bGo" + _SEP + r"(?i:" + _GO_NEIGHBORS + r")\b"
    r"|(?i:\b(?:using|in|with)\s+)Go\b(?![\w\-])"
)

# bare "Excel" is case-sensitive (avoids the verb "excel"); phrases are not.
_EXCEL_PATTERN = (
    r"(?<![A-Za-z0-9])(?:Excel|(?i:microsoft\s+excel|ms\s+excel|advanced\s+excel))"
    r"(?![A-Za-z0-9])"
)


TAXONOMY: list[Skill] = [
    # --- Programming languages ---
    Skill("Python", "language", ("Python",)),
    Skill("R", "language", pattern=_R_PATTERN),
    Skill("SQL", "language", ("SQL",)),
    Skill("NoSQL", "language", ("NoSQL",)),
    Skill("Java", "language", ("Java",)),
    Skill("Scala", "language", ("Scala",)),
    Skill("JavaScript", "language", ("JavaScript",)),
    Skill("TypeScript", "language", ("TypeScript",)),
    Skill("C++", "language", ("C++",)),
    Skill("C#", "language", ("C#",)),
    Skill("Go", "language", pattern=_GO_PATTERN),
    Skill("Rust", "language", ("Rust",)),
    Skill("Julia", "language", ("Julia",)),
    Skill("MATLAB", "language", ("MATLAB",)),
    Skill("SAS", "language", ("SAS",)),
    Skill("SPSS", "language", ("SPSS",)),
    Skill("Stata", "language", ("Stata",)),

    # --- Databases / warehouses ---
    Skill("PostgreSQL", "database", ("PostgreSQL", "Postgres")),
    Skill("MySQL", "database", ("MySQL",)),
    Skill("SQL Server", "database", ("SQL Server", "MSSQL", "T-SQL")),
    Skill("Oracle", "database", ("Oracle",)),
    Skill("MongoDB", "database", ("MongoDB", "Mongo")),
    Skill("Redis", "database", ("Redis",)),
    Skill("Cassandra", "database", ("Cassandra",)),
    Skill("Snowflake", "database", ("Snowflake",)),
    Skill("BigQuery", "database", ("BigQuery", "Big Query")),
    Skill("Redshift", "database", ("Redshift",)),
    Skill("Elasticsearch", "database", ("Elasticsearch", "Elastic Search")),

    # --- Cloud ---
    Skill("AWS", "cloud", ("AWS", "Amazon Web Services")),
    Skill("Azure", "cloud", ("Azure", "Microsoft Azure")),
    Skill("GCP", "cloud", ("GCP", "Google Cloud Platform", "Google Cloud")),

    # --- Big data / data engineering ---
    Skill("Spark", "big_data", ("Apache Spark", "PySpark", "Spark")),
    Skill("Hadoop", "big_data", ("Hadoop",)),
    Skill("Kafka", "big_data", ("Kafka",)),
    Skill("Hive", "big_data", ("Hive",)),
    Skill("Airflow", "big_data", ("Airflow",)),
    Skill("dbt", "big_data", ("dbt",)),
    Skill("Databricks", "big_data", ("Databricks",)),
    Skill("Flink", "big_data", ("Flink",)),

    # --- ML / DS libraries ---
    Skill("scikit-learn", "ml_library", ("scikit-learn", "scikit learn", "sklearn")),
    Skill("TensorFlow", "ml_library", ("TensorFlow", "Tensor Flow")),
    Skill("PyTorch", "ml_library", ("PyTorch",)),
    Skill("Keras", "ml_library", ("Keras",)),
    Skill("pandas", "ml_library", ("pandas",)),
    Skill("NumPy", "ml_library", ("NumPy",)),
    Skill("SciPy", "ml_library", ("SciPy",)),
    Skill("XGBoost", "ml_library", ("XGBoost",)),
    Skill("spaCy", "ml_library", ("spaCy",)),
    Skill("Hugging Face", "ml_library", ("Hugging Face", "HuggingFace", "transformers")),

    # --- BI / visualization ---
    Skill("Tableau", "viz_bi", ("Tableau",)),
    Skill("Power BI", "viz_bi", ("Power BI", "PowerBI")),
    Skill("Looker", "viz_bi", ("Looker",)),
    Skill("Qlik", "viz_bi", ("Qlik", "QlikView", "Qlik Sense")),
    Skill("Excel", "viz_bi", pattern=_EXCEL_PATTERN),
    Skill("Matplotlib", "viz_bi", ("Matplotlib",)),
    Skill("Plotly", "viz_bi", ("Plotly",)),
    Skill("Seaborn", "viz_bi", ("Seaborn",)),

    # --- DevOps / tooling ---
    Skill("Docker", "devops", ("Docker",)),
    Skill("Kubernetes", "devops", ("Kubernetes", "K8s")),
    Skill("Git", "devops", ("Git", "GitHub", "GitLab")),
    Skill("Terraform", "devops", ("Terraform",)),
    Skill("Linux", "devops", ("Linux",)),

    # --- Methods / concepts ---
    Skill("Machine Learning", "method", ("Machine Learning",)),
    Skill("Deep Learning", "method", ("Deep Learning",)),
    Skill("NLP", "method", ("NLP", "Natural Language Processing")),
    Skill("Computer Vision", "method", ("Computer Vision",)),
    Skill("Statistics", "method", ("Statistics", "Statistical")),
    Skill("A/B Testing", "method", ("A/B Testing", "A/B Test", "AB Testing")),
    Skill("ETL", "method", ("ETL", "ELT")),
    Skill("Data Visualization", "method", ("Data Visualization", "Data Viz")),
    Skill("Time Series", "method", ("Time Series",)),
    Skill("Data Modeling", "method", ("Data Modeling", "Data Modelling")),
]
