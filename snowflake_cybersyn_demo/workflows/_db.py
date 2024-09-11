import json
from typing import Any, Dict, List

from snowflake.sqlalchemy import URL
from sqlalchemy import create_engine, text

from snowflake_cybersyn_demo.utils import load_from_env

snowflake_user = load_from_env("SNOWFLAKE_USERNAME")
snowflake_password = load_from_env("SNOWFLAKE_PASSWORD")
snowflake_account = load_from_env("SNOWFLAKE_ACCOUNT")
snowflake_role = load_from_env("SNOWFLAKE_ROLE")

CANDIDATE_LIST_SQL_QUERY_TEMPLATE = """
SELECT DISTINCT att.product,
FROM cybersyn.bureau_of_labor_statistics_price_timeseries AS ts
JOIN cybersyn.bureau_of_labor_statistics_price_attributes AS att
    ON (ts.variable = att.variable)
WHERE ts.date >= '2021-01-01'
  AND att.report = 'Average Price'
  AND att.product ILIKE '{good}%';
"""


TIMESERIES_SQL_QUERY_TEMPLATE = """
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


GOVT_ESSENTIALS_SQL_QUERY_TEMPLATE = """
SELECT
    ts.date as date,
    ts.variable_name,
    ts.value as value
FROM cybersyn.datacommons_timeseries AS ts
JOIN cybersyn.geography_index AS geo ON (ts.geo_id = geo.geo_id)
WHERE geo.geo_name = '{city}'
  AND geo.level IN ('City')
  AND ts.variable_name ILIKE '{stats_variable}%'
  AND date >= '2015-01-01'
ORDER BY date;
"""


SQL_QUERY_TEMPLATE = """
SELECT DISTINCT
       ts.variable_name
FROM cybersyn.datacommons_timeseries AS ts
JOIN cybersyn.geography_index AS geo ON (ts.geo_id = geo.geo_id)
WHERE geo.geo_name = '{city}'
  AND geo.level IN ('City')
  AND date >= '2015-01-01';
"""


def get_list_of_statistical_variables(city: str) -> List[str]:
    """Returns a list of statistical variables that closely resemble the query.

    The list of statistical vars is represented as a string separated by '\n'.
    """
    query = SQL_QUERY_TEMPLATE.format(city=city)
    url = URL(
        account=snowflake_account,
        user=snowflake_user,
        password=snowflake_password,
        database="GOVERNMENT_ESSENTIALS",
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
    return [f"{ix+1}. {str(el[0])}" for ix, el in enumerate(results)]


def get_time_series_of_statistic_variable(
    city: str, stats_variable: str
) -> str:
    """Create a time series of a specified stats variable."""
    query = GOVT_ESSENTIALS_SQL_QUERY_TEMPLATE.format(
        city=city, stats_variable=stats_variable
    )
    url = URL(
        account=snowflake_account,
        user=snowflake_user,
        password=snowflake_password,
        database="GOVERNMENT_ESSENTIALS",
        schema="CYBERSYN",
        warehouse="COMPUTE_WH",
        role=snowflake_role,
    )

    engine = create_engine(url)
    try:
        connection = engine.connect()
        results = connection.execute(text(query))
    except Exception:
        raise
    finally:
        connection.close()

    # process
    results = [
        {"variable": str(el[1]), "date": str(el[0]), "value": str(el[2])}
        for el in results
    ]
    results_str = json.dumps(results, indent=4)

    return results_str


def get_list_of_candidate_goods(good: str) -> List[str]:
    """Returns a list of goods that exist in the database.

    The list of goods is represented as a string separated by '\n'."""
    query = CANDIDATE_LIST_SQL_QUERY_TEMPLATE.format(good=good)
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

    return [f"{ix+1}. {str(el[0])}" for ix, el in enumerate(results)]


def get_time_series_of_good(good: str) -> str:
    """Create a time series of the average price paid for a good nationwide starting in 2021."""
    query = TIMESERIES_SQL_QUERY_TEMPLATE.format(good=good)
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


def perform_date_value_aggregation(json_str: str) -> List[Dict[str, Any]]:
    """Perform value aggregation on the time series data."""
    timeseries_data = json.loads(json_str)
    variable = timeseries_data[0]["variable"]

    new_time_series_data: Dict[str, List[float]] = {}
    for el in timeseries_data:
        date = el["date"]
        value = el["value"]
        if date in new_time_series_data:
            new_time_series_data[date].append(float(value))
        else:
            new_time_series_data[date] = [float(value)]

    reduced_time_series_data = [
        {
            "variable": variable,
            "date": date,
            "value": sum(values) / len(values),
        }
        for date, values in new_time_series_data.items()
    ]

    return reduced_time_series_data


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
