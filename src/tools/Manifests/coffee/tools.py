from langchain_core.tools import tool


@tool
def process_image(image_data: str):
    """this tool expects binary image, processes the image, analyzes appearance, and returns a greeting.
       Greeting should be dynamic based on clothing, age, glasess, gender and could be anything else.
    """
    print(image_data[0:100])
    return ["greeting"]

@tool()
def ask_entities():
    """
    {ask_entities}
    """
    return "entities collection"

@tool
def confirm_coffee_information(
    roast_level: str,
    flavor_profile: str,
    brewing_method: str,
    price_range: str,
    origin: str,
    occasion: str,
):
    """{confirm_coffee_information}"""
    collected_human_data = {
        "roast_level": roast_level,
        "flavor_profile": flavor_profile,
        "brewing_method": brewing_method,
        "price_range": price_range,
        "origin": origin,
        "occasion": occasion,
    }
    return [f"ask to confirm on collected information : {collected_human_data}"]


@tool
def confirm_coffee_information(roast_level: str, flavor_profile: str, brewing_method: str, price_range: str, origin: str, occasion: str):
    """
    {confirm_coffee_information}
    """
    collected_human_data = {
      "roast_level": roast_level,
      "flavor_profile": flavor_profile, 
      "brewing_method": brewing_method,
      "price_range": price_range,
      "origin": origin,
      "occasion": occasion
    }
    return [f"ask to confirm on collected information : {collected_human_data}"]

