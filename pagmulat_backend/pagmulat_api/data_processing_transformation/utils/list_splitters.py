def make_list_from_commas(value):
    if isinstance(value, str):
        return [item.strip() for item in value.split(',') if item.strip()]
    return []
