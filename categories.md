---
layout: default
title: Categories
---
<ul class="list-unstyled">
{% for cat in site.categories %} 
    <a name="{{ cat[0] }}"></a>
    <h2>{{ cat[0] }}({{ cat[1].size }})</h2> 
    {% for post in cat[1] %} 
        <li><span>{{ post.date | date_to_string }} &raquo;</span><a class="post-link" href="{{ post.url }}">{{ post.title }}</a></li>
    {% endfor %} 
{% endfor %} 
</ul>

