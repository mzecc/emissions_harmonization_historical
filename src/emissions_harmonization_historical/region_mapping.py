"""Here we specify functions used to generate the 'region_df.csv' from the common-definitions repository."""

from pathlib import Path
import os
from nomenclature import countries
from nomenclature.definition import DataStructureDefinition
from nomenclature.processor import RegionProcessor
import pandas as pd

from emissions_harmonization_historical.constants import DATA_ROOT

# define output paths+names
region_file = DATA_ROOT / Path("region_df.csv")
mappings_file = DATA_ROOT / Path("mappings_df.csv")


# Set working directory
working_directory = Path(__file__).parent.parent.parent.parent / "common-definitions"  # Directory of the current script
os.chdir(working_directory)


# a DSD stores the definitions info for regions, variables, scenarios
dsd = DataStructureDefinition("definitions")

# dsd.region stores a dictionary of <region name>: <RegionCode object>
# a RegionCode object stores the info for a region defined in the YAML files, such as the model (
region_df = pd.DataFrame(
    [(r.name, r.hierarchy, r.countries, r.iso3_codes) for r in dsd.region.values()],
    columns=["name", "hierarchy", "countries", "iso3"]
)


# fill currently empty iso3 column
# Function to fetch ISO3 codes
def get_iso3_list(country_list):
    if not country_list:
        return None
    iso3_list = []
    for country in country_list:
        try:
            iso3 = countries.get(name=country)
            if iso3:
                iso3_list.append(iso3.alpha_3)
            else:
                iso3_list.append(None)  # If no match found
        except Exception:
            iso3_list.append(None)
    return iso3_list


# fill the iso3 column
region_df["iso3"] = region_df["countries"].apply(get_iso3_list)

# the RegionProcessor creates the mappings object
# the mappings are defined at the model level in a RegionAggregationMapping object
# each RAM contains, among other things, the list of "common regions" (regions resulting
# from aggregation) and their constituents
rp = RegionProcessor.from_directory("mappings", dsd)
rows = []
for ram in rp.mappings.values():
    rows.extend(
        [
            (
                ram.model,
                common_region.name,
                [ram.rename_mapping[nr] for nr in common_region.constituent_regions],
            )
            for common_region in ram.common_regions
        ]
    )
mappings_df = pd.DataFrame(
    rows,
    columns=["model(s)", "common_region", "constituent_regions"],
)


region_df.to_csv(region_file)
#mappings_df.to_csv(mappings_file)