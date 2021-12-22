{{ fullname.split('.')[-1] | escape | underline}}

{% for idx in range(fullname.split('.')[:-1]|length) %}
   :py:mod:{{'`~'+ '.'.join(fullname.split('.')[:(idx+1)]) + '`'}} .
{%- endfor %}
   :py:mod:{{'`~'+ fullname + '`'}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :special-members: __call__, __add__, __mul__

   {% block methods %}
   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :nosignatures:
   {% for item in methods %}
      {%- if item not in inherited_members %}
         {%- if not item.startswith('_') %}
            ~{{ name }}.{{ item }}
         {%- endif -%}
      {%- endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% for item in attributes %}
      {%- if item not in inherited_members %}
         {%- if not item.startswith('_') %}
            ~{{ name }}.{{ item }}
         {%- endif -%}
      {%- endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}
