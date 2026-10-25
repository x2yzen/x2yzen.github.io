---
layout: post
title: "2025 AI Retrospective: From Agents to Applications"
description: "A look back at agents, reasoning, context engineering, the applications that worked, and the ones that did not."
date: 2025-12-23 11:00:00 -0800
permalink: /posts/2025-annual-review/en/
categories: [language_model]
tags: [llm, product]
lang: en
math: false
translations:
  current: en
  zh: /posts/2025-annual-review/
  en: /posts/2025-annual-review/en/
---

{% include language-switch.html %}

## agents

At the start of the year, it was still hard to say exactly what an *agent* was. By the end of it, a simple definition had become common: **an agent is an LLM using tools in a loop**. That also captures the two main directions of the year: models that are better at using tools, and tools that are easier to access through interfaces such as MCP.

## what worked

At the model level, the biggest developments were the **thinking paradigm** and the wider use of **reinforcement learning**. Both were marked by the arrival of o1 at the end of 2024. For a problem that takes several steps, letting an autoregressive model produce intermediate reasoning before its answer is an intuitive idea. During 2025, the major models adopted this approach and pushed up scores on reasoning benchmarks. Reinforcement learning also moved beyond preference alignment. Training on verifiable tasks in STEM and coding greatly improved performance in those areas, with some gains carrying over to other reasoning tasks.

In engineering, the year's central theme was **context engineering**. A language model samples from a distribution conditioned on its input. The application has to assemble enough relevant context for the task; the model then has to make good use of it. The first version of this work was prompt engineering: writing a system prompt that explained the intent and gave examples. Now much of the context comes from tools: retrieved documents, database rows, code output, and tool descriptions. These pieces are more fragmented than a prompt. Meanwhile, sparse attention and long-context performance still fall short of what we need. As a context window fills up, returns diminish. The hard question is what to put in it, when to put it there, and what to leave out. Better tool design, offloading inactive material to files, keeping useful memories, and isolating work in subagents are all ways to manage that limited space. Context management has become one of the main challenges in building AI applications.

![Diagram about context engineering](/assets/images/2025-12-23-2025-annual-review/2.png)

*Source: Claude engineering blog*

At the product level, agents became genuinely useful in **search** and **coding**. Deep research gives search more depth than a conventional summary without asking much more of the user. It strengthens an already good product fit, and I expect it to change search engines substantially. Claude Code, released in the first half of the year, combined a strong coding model with a practical agent framework. It became useful for understanding a repository and writing fairly self-contained scripts. Changes to a large, existing codebase remain harder, perhaps because the agent does not yet gather enough of the project's requirements and history. That level of capability is not a general replacement for software engineers. Work centered on scripts, such as some ETL and exploratory analysis, may change sooner. It has also let people with product or design backgrounds build ideas into something between a mockup and a usable product.

## what didn't

Outside those few cases, many AI products still produce less value than the effort put into them.

At the model level, **agentic capability**—especially generalizing reasoning and following instructions—improved less than I expected. Flagship models climbed benchmarks such as AIME, GPQA Diamond, and HLE, yet can still get tangled in the arithmetic of an ordinary mortgage choice. Reasoning learned in verifiable training environments has not transferred smoothly to everyday tasks. Near the limits of a long context window, models still forget facts, miss instructions, or mix up people and claims. These failures limit what products can reliably do.

**Fine-tuning**, so popular a year earlier, also lost momentum. Larger and more complicated models make it expensive to modify them. More fundamentally, an end-to-end reinforcement-learning pipeline is hard to build for real applications. Its *states* and *rewards* are often vague, delayed, or sparse. Falling technical costs will help, but many desired applications do not fit the training setup's implicit assumptions: that a state can be described and feedback can be measured. RL can solve chess, Go, and contest math because those worlds are formal. Everyday work often is not. Two years ago it seemed possible that language alone would be a general key; so far it looks more like half a step.

At the product level, the **MCP tool market** has not taken off: many people are building tools, but relatively few are using them. It may need more AI products that are productive in real workflows. **AI companionship** cooled quickly as well; emotional value may be hard to sustain when the other side does not take part in social life. Startups that apply AI to broad, valuable tasks such as search and coding, but own no essential data or workflow, are being squeezed by incumbents and foundation-model companies. The window for getting there first is narrowing. Finding a real need, and owning a place in the data or workflow around it, matters more.

![MCP meme](/assets/images/2025-12-23-2025-annual-review/1.png)

*MCP meme*

## the market

The outlook for talent in foundation models is uncertain. More people now argue that simple scaling is giving way to research-driven progress. If the next major leap requires a new architecture or training idea, creative newcomers may matter more. From outside the top research labs, it is hard to say which direction will work.

Applications are easier to reason about. Large-company engineers will benefit as AI product frameworks and infrastructure mature; learning to use them is not especially hard. The scarce skill is understanding a domain well enough to **make its states, resources, and goals explicit**, then giving a model the right context at the right time. That alone can produce a leading application. A team that can also define a measurable environment and dataset for end-to-end reinforcement learning may do something more unusual.

I expect startups to become more vertical. Standard, high-value general tasks—search, coding, assistants, spreadsheets—are likely to be absorbed by large platforms. A founding team should start with a field whose business model it understands, then use AI to lower costs or create a new experience. The better opportunities may lie in industries with deep existing barriers, such as healthcare and insurance, or in creative fields such as animation and games. People who know AI best often know little about those markets. That gap is both a limitation and an opening.
