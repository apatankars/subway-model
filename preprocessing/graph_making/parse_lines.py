import re

def split_by_slash(text):
  """
  split a string by forward slashes '/' and returns a list of substrings
  no slash is present --> return a list containing the original string

  input string

  return: list of strings arounds the slashes
  """
  if not isinstance(text, str):
    return []
  return text.split('/')


# Using Regex (alternative)
def extract_last_paren_content_regex(string):
    '''extract content from the last parentheses statement in string'''
    if not isinstance(string, str):
        return None
    # match last occurrence of string in parentheses
    match = re.search(r'\(([^()]*)\)[^()]*$', string)
    return match.group(1) if match else None

def lines_from_station_complex(complex_str):
    '''
    extract the station names and a list of lines from station_complex

    input:
        complex_str: station name string, e.g.,
                           'cathedral pkwy (1,2,3)' or 'first (A)/second (B)'.

    return:
        (base_name, lines_list)
            base_name: string station name part, keeping slashes, removing lines
            lines_list: list of strings representing the lines,
                              extracted from the last parentheses.
                             empty list if no lines found.
    '''
    if not isinstance(complex_str, str):
        return None, []

    lines_content = extract_last_paren_content_regex(complex_str)

    if lines_content is not None:
        # lines from last paren
        lines_list = [line.strip() for line in lines_content.split(',') if line.strip()]

        # start index of last paren
        last_paren_start_index = complex_str.rfind(f'({lines_content})')
        if last_paren_start_index != -1:
            base_name = complex_str[:last_paren_start_index].strip()
        else:
            base_name = complex_str.strip()
    else:
        # no lines
        lines_list = []
        base_name = complex_str.strip()

    return base_name, lines_list

# # tests
# test1 = 'cathedral pkwy (1,2,3)'
# test2 = 'first (A)/second (B)'
# test3 = 'Lexington Av/63 St (F,Q)'
# test4 = 'No lines here'
# test5 = 'Single Line (A)'
# test6 = 'cathedral parkway (110 st) (1)'
# test7 = None
# test8 = 'Times Sq-42 St (N,Q,R,W,S,1,2,3,7)'

# print(f"'{test1}' -> {lines_from_station_complex(test1)}")
# print(f"'{test2}' -> {lines_from_station_complex(test2)}")
# print(f"'{test3}' -> {lines_from_station_complex(test3)}")
# print(f"'{test4}' -> {lines_from_station_complex(test4)}")
# print(f"'{test5}' -> {lines_from_station_complex(test5)}")
# print(f"'{test6}' -> {lines_from_station_complex(test6)}")
# print(f"'{test7}' -> {lines_from_station_complex(test7)}")
# print(f"'{test8}' -> {lines_from_station_complex(test8)}")

