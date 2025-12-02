# Data Sources

This directory contains input data for the renewable energy opposition analysis pipeline.

## EIA Data

Generator-level data from the U.S. Energy Information Administration Form EIA-860.

| File | Description | Source |
|------|-------------|--------|
| `eia_plants_2022.csv` | Plant characteristics (location, capacity, ownership) | [EIA Electricity Data](https://www.eia.gov/electricity/data.php) |
| `eia_generation_2022.csv` | Annual generation by plant | [EIA Electricity Data](https://www.eia.gov/electricity/data.php) |

Download from EIA: Select "Annual" data under "Generator-level data" section.

## State Mappings

| File | Description |
|------|-------------|
| `us_state_abbreviations.json` | State name to abbreviation mapping |
| `us_abbreviations_to_state.json` | Abbreviation to state name mapping |

## Demographic Data

Census tract-level demographic and environmental justice data.

| File | Description | Source |
|------|-------------|--------|
| `demographic_data/1.0-shapefile-codebook.zip` | EJScreen shapefile with Census demographics | EPA EJScreen (no longer hosted; archived at [pedp-ejscreen.azurewebsites.net](https://pedp-ejscreen.azurewebsites.net/)) |
| `demographic_data/1.0-shapefile-codebook/1.0-codebook.csv` | Variable definitions | EPA EJScreen |
| `demographic_data/1.0-shapefile-codebook/usa/columns.csv` | Column descriptions for shapefile | EPA EJScreen |
| `demographic_data/cb_2023_us_state_20m.zip` | State boundary shapefile | [U.S. Census TIGER/Line](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html) |

The EJScreen dataset joins EPA environmental indicators with American Community Survey demographic data at the Census tract level.
