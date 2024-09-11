import json
from typing import Any, Dict, List

from snowflake.sqlalchemy import URL
from sqlalchemy import create_engine, text

from snowflake_cybersyn_demo.utils import load_from_env

snowflake_user = load_from_env("SNOWFLAKE_USERNAME")
snowflake_password = load_from_env("SNOWFLAKE_PASSWORD")
snowflake_account = load_from_env("SNOWFLAKE_ACCOUNT")
snowflake_role = load_from_env("SNOWFLAKE_ROLE")
localhost = load_from_env("LOCALHOST")

SQL_QUERY_TEMPLATE = """
SELECT ts.date,
       att.variable_name,
       ts.value
FROM cybersyn.bureau_of_labor_statistics_price_timeseries AS ts
JOIN cybersyn.bureau_of_labor_statistics_price_attributes AS att
    ON (ts.variable = att.variable)
WHERE ts.date >= '2021-01-01'
  AND att.report = 'Average Price'
  AND att.product ILIKE '{good}%'
ORDER BY date;
"""


def get_time_series_of_good(good: str) -> str:
    """Create a time series of the average price paid for a good nationwide starting in 2021."""
    query = SQL_QUERY_TEMPLATE.format(good=good)
    url = URL(
        account=snowflake_account,
        user=snowflake_user,
        password=snowflake_password,
        database="FINANCIAL__ECONOMIC_ESSENTIALS",
        schema="CYBERSYN",
        warehouse="COMPUTE_WH",
        role=snowflake_role,
    )

    engine = create_engine(url)
    try:
        connection = engine.connect()
        results = connection.execute(text(query))
    finally:
        connection.close()

    # process
    results = [
        {"good": str(el[1]), "date": str(el[0]), "price": str(el[2])}
        for el in results
    ]
    results_str = json.dumps(results, indent=4)

    return results_str


def perform_price_aggregation(json_str: str) -> List[Dict[str, Any]]:
    """Perform price aggregation on the time series data."""
    timeseries_data = json.loads(json_str)
    good = timeseries_data[0]["good"]

    new_time_series_data: Dict[str, List[float]] = {}
    for el in timeseries_data:
        date = el["date"]
        price = el["price"]
        if date in new_time_series_data:
            new_time_series_data[date].append(float(price))
        else:
            new_time_series_data[date] = [float(price)]

    reduced_time_series_data = [
        {"good": good, "date": date, "price": sum(prices) / len(prices)}
        for date, prices in new_time_series_data.items()
    ]

    return reduced_time_series_data
