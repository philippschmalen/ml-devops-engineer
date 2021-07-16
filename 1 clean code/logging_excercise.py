"""
Standard setup for logging
"""
import logging

logging.basicConfig(
    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
    datefmt='%d-%m %H:%M',
    filename='log/logging_excercise.log',
    level=logging.INFO
)

def sum_vals(a, b):
    '''
    Args:
        a: (int)
        b: (int)
    Return:
        a + b (int)
    '''
    try:
        assert isinstance(a, int), "Input a should be int"
        assert isinstance(b, int), "Input b should be int"
        result = a+b
        logging.info("SUCCESS: sum calculated")
        return result
    except AssertionError:
        logging.error("Input does not match expected type.")


if __name__ == "__main__":
    sum_vals('no', 'way')
    sum_vals(4, 5)
