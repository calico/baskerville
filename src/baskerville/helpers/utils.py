import pickle


def load_extra_options(options_pkl_file, options):
    """
    Args:
        options_pkl_file: option file
        options: existing options from command line
    Returns:
        options: updated options
    """
    options_pkl = open(options_pkl_file, "rb")
    new_options = pickle.load(options_pkl)
    new_option_attrs = vars(new_options)
    # Assuming 'options' is the existing options object
    # Update the existing options with the new attributes
    for attr_name, attr_value in new_option_attrs.items():
        setattr(options, attr_name, attr_value)
    options_pkl.close()
    return options
