{{ fullname.split('.')[-1] | escape | underline }}

{% for idx in range(fullname.split('.')[:-1]|length) %}
   :py:mod:{{'`~'+ '.'.join(fullname.split('.')[:(idx+1)]) + '`'}} .
{%- endfor %}
   :py:mod:{{'`~'+ fullname + '`'}}

.. currentmodule:: {{ module }}

.. auto{{ objtype }}:: {{ objname }}
