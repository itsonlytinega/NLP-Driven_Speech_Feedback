"""
Custom template filters for mathematical operations.
"""

from django import template

register = template.Library()


@register.filter(name='div')
def div(value, arg):
    """
    Divides the value by the argument.
    
    Usage: {{ value|div:divisor }}
    """
    try:
        value = float(value)
        arg = float(arg)
        if arg == 0:
            return 0
        return value / arg
    except (ValueError, TypeError):
        return 0


@register.filter(name='mul')
def mul(value, arg):
    """
    Multiplies the value by the argument.
    
    Usage: {{ value|mul:multiplier }}
    """
    try:
        value = float(value)
        arg = float(arg)
        return value * arg
    except (ValueError, TypeError):
        return 0


@register.filter(name='subtract')
def subtract(value, arg):
    """
    Subtracts the argument from the value.
    
    Usage: {{ value|subtract:number }}
    """
    try:
        value = float(value)
        arg = float(arg)
        return value - arg
    except (ValueError, TypeError):
        return 0


@register.filter(name='add_custom')
def add_custom(value, arg):
    """
    Adds the argument to the value.
    
    Usage: {{ value|add_custom:number }}
    """
    try:
        value = float(value)
        arg = float(arg)
        return value + arg
    except (ValueError, TypeError):
        return 0


@register.filter(name='percentage')
def percentage(value, arg):
    """
    Calculates percentage of value out of arg.
    
    Usage: {{ value|percentage:total }}
    """
    try:
        value = float(value)
        arg = float(arg)
        if arg == 0:
            return 0
        return (value / arg) * 100
    except (ValueError, TypeError):
        return 0


