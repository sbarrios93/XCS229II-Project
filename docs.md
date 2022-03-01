---
title: Hello
header-includes:
    - \renewcommand{\vec}[1]{\mathbf{#1}}
    -
---


```mermaid
graph TD
A(hello) -- Hi --> B(world)
B --> C
C --> D(This is a super long text of nothing really im just )
D --> A
D --> F(This is a rather long text)
C --> F --> G
```

(@)  My first example will be numbered (1).
(@)  My second example will be numbered (2).

Explanation of examples.

(@)  My third example will be numbered (3).
Numbered examples can be labeled and referred to elsewhere in the document:

(@good)  This is a good example.

As (@good) illustrates, ...



The gravitational force

$$\vec{g}$$

The gravitational force

$$\mathbf{g}$$

And with some code:

~~~{.cpp .numberLines startFrom="1"}
class A {};
~~~