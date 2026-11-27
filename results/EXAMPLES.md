# Examples: held-out puzzles, base vs main adapter vs control

Greedy decoding, 512-token limit, prompt v2. Picked by rule (`scripts/make_examples.py`: the first 2 puzzles by id in each category), not by hand. All 500 answers per model are in `results/kaggle/*/eval/*.jsonl`.

## Base was cut off, main finished and got it right (the most common kind of gain) (75 of 500)

**heldout-n5-00006**: numbers [49, 89, 62, 41, 18], target 17

- base: cut off at 512 tokens, no answer
- main: `(89 - 62 - 41) + 49 - 18` — correct (443 tokens)
- control: cut off at 512 tokens, no answer

<details><summary>Main adapter's full answer</summary>

```text
To create an equation that equals 17 using the numbers 49, 89, 62, 41, and 18 exactly once, we can try the following:

Start by subtracting two larger numbers to get a smaller value:

$$
89 - 62 = 27
$$

Now subtract another number:

$$
27 - 41 = -14
$$

Add the remaining numbers:

$$
-14 + 49 + 18 = 53
$$

This is too large. Let's try a different approach.

Try:

$$
89 - 49 = 40
$$

$$
40 - 41 = -1
$$

$$
-1 + 62 + 18 = 79
$$

Still too large. Try:

$$
62 - 49 = 13
$$

$$
13 + 18 = 31
$$

$$
31 - 89 = -58
$$

$$
-58 + 41 = -17
$$

Not quite. Try:

$$
89 - 49 = 40
$$

$$
40 - 62 = -22
$$

$$
-22 + 41 = 19
$$

$$
19 - 18 = 1
$$

Still not 17. Try:

$$
89 - 62 = 27
$$

$$
27 - 41 = -14
$$

$$
-14 + 49 = 35
$$

$$
35 - 18 = 17
$$

This works!

So the equation is:

$$
(89 - 62 - 41) + 49 - 18 = 17
$$

Final answer:

<answer>(89 - 62 - 41) + 49 - 18</answer>
```

</details>

**heldout-n5-00012**: numbers [51, 6, 74, 15, 2], target 34

- base: cut off at 512 tokens, no answer
- main: `(74 - 51) + 15 - 6 + 2` — correct (267 tokens)
- control: `(74 - 51) + (15 - 6) + 2` — correct (427 tokens)

<details><summary>Main adapter's full answer</summary>

```text
To create an equation that equals 34 using the numbers 51, 6, 74, 15, and 2 exactly once, we can try the following:

Start by subtracting 74 and 15 from 51:  
51 - 74 = -23  
-23 - 15 = -38  

Now, add 6 and 2:  
6 + 2 = 8  

Now, add -38 and 8:  
-38 + 8 = -30  

This is not correct. Let's try a different approach.

Try:  
(74 - 51) = 23  
23 + 15 = 38  
38 - 6 = 32  
32 + 2 = 34  

So the equation is:  
(74 - 51) + 15 - 6 + 2 = 34  

Now, check that we used each number exactly once: 74, 51, 15, 6, 2 — yes.  

Final equation:  
<answer>(74 - 51) + 15 - 6 + 2</answer>
```

</details>

## Both right (46 of 500)

**heldout-n5-00000**: numbers [30, 77, 61, 85, 49], target 82

- base: `(85 - 61) + (77 - 49) + 30` — correct (229 tokens)
- main: `30 + (85 - 61) + (77 - 49)` — correct (241 tokens)
- control: `(85 - 61) + (77 - 49) + 30` — correct (216 tokens)

<details><summary>Main adapter's full answer</summary>

```text
To create an equation that equals 82 using the numbers 30, 77, 61, 85, and 49 exactly once, we can try the following:

Start by considering subtraction and addition:

$$
85 - 77 = 8
$$
$$
61 - 49 = 12
$$
$$
30 + 8 + 12 = 50 \quad \text{(too low)}
$$

Try a different approach:

$$
85 - 61 = 24
$$
$$
77 - 49 = 28
$$
$$
30 + 24 + 28 = 82
$$

So the equation is:

$$
30 + (85 - 61) + (77 - 49) = 82
$$

Now simplify:

$$
30 + 24 + 28 = 82
$$

Final equation:

<answer>30 + (85 - 61) + (77 - 49)</answer>
```

</details>

**heldout-n5-00016**: numbers [49, 8, 87, 90, 90], target 36

- base: `(90 - 87) + (90 - 49 - 8)` — correct (294 tokens)
- main: `(90 - 87) + (90 - 49) - 8` — correct (191 tokens)
- control: `(90 - 87) + (90 - 49) - 8` — correct (146 tokens)

<details><summary>Main adapter's full answer</summary>

```text
To create an equation that equals 36 using the numbers 49, 8, 87, 90, 90 exactly once, we can try the following:

Start by considering the largest numbers and how they can be reduced. Notice that 90 - 87 = 3, and 90 - 49 = 41. However, we need to reach 36.

Try this combination:

$$
(90 - 87) + (90 - 49) - 8 = 3 + 41 - 8 = 36
$$

This uses all the numbers exactly once: 90, 87, 90, 49, and 8.

So the final equation is:

<answer>(90 - 87) + (90 - 49) - 8</answer>
```

</details>

## Base right, main wrong (50 of 500)

**heldout-n5-00005**: numbers [36, 44, 38, 43, 18], target 91

- base: `(43 + 38 + 18) - (44 - 36)` — correct (335 tokens)
- main: cut off at 512 tokens, no answer
- control: cut off at 512 tokens, no answer

<details><summary>Main adapter's full answer</summary>

```text
To create an equation that equals 91 using the numbers 36, 44, 38, 43, and 18 exactly once, we can try the following:

Start by adding some numbers and subtracting others:

$$
44 + 38 + 18 = 100
$$

Now subtract 9 to get 91:

$$
100 - 9 = 91
$$

We need to form 9 from the remaining numbers: 36 and 43. Notice that:

$$
43 - 36 = 7
$$

But we need 9, not 7. Let's try a different approach.

Try:

$$
44 + 36 = 80
$$

Now we need 11 more. Try:

$$
43 - 38 = 5
$$
$$
18 - 5 = 13
$$

Too much. Try:

$$
43 + 18 = 61
$$
$$
38 + 36 = 74
$$
$$
74 - 61 = 13
$$

Still not right. Try:

$$
44 + 38 = 82
$$
$$
43 + 18 = 61
$$
$$
82 - 61 = 21
$$

Too much. Try:

$$
44 + 36 = 80
$$
$$
43 - 18 = 25
$$
$$
80 + 25 = 105
$$

Too much. Try:

$$
44 + 38 = 82
$$
$$
43 - 18 = 25
$$
$$
82 - 25 = 57
$$

Too little. Try:

$$
44 + 36 = 80
$$
$$
43 - 38 = 5
$$
$$
80 + 5 = 85
$$

Still not 91. Try:

$$
44 + 38 = 82
$$
$$
43 + 18 = 61
$$
$$
82 + 61 = 143
$$

Too much. Try
```

</details>

**heldout-n5-00015**: numbers [43, 1, 9, 88, 98], target 24

- base: `(43 - 9) - (98 - 88) * 1` — correct (505 tokens)
- main: cut off at 512 tokens, no answer
- control: cut off at 512 tokens, no answer

<details><summary>Main adapter's full answer</summary>

```text
To create an equation that equals 24 using the numbers 43, 1, 9, 88, and 98 exactly once, we can try the following:

First, note that 88 and 98 are large numbers, so we need to subtract or divide them to bring the value down. Let's try:

$$
(98 - 88) = 10
$$

Now we have 10, and we need to use 43, 1, and 9. Let's try:

$$
43 - 9 = 34
$$

Now we have 34 and 10. Try:

$$
34 - 10 = 24
$$

Now we have used 98, 88, 43, and 9. The only number left is 1. We need to incorporate it. Let's try:

$$
(98 - 88) + (43 - 9) - 1 = 24
$$

Check the arithmetic:

$$
(98 - 88) = 10 \\
(43 - 9) = 34 \\
10 + 34 = 44 \\
44 - 1 = 43 \quad \text{(Not 24)}
$$

Try a different approach:

$$
(98 - 88) = 10 \\
(43 - 9) = 34 \\
34 - 10 = 24 \\
24 - 1 = 23 \quad \text{(Still not 24)}
$$

Try:

$$
(98 - 88) = 10 \\
(43 - 9 - 1) = 33 \\
33 - 10 = 23 \quad \text{(Still not 24)}
$$

Try:

$$
(98 - 88) = 10 \\
(43 - 9) = 34 \\
34 - 10 = 24 \\
24 + 1 = 25 \quad \text{(Still not 24)}
$$

Try:

$$
(98 - 88) = 10 \\
(43 - 9 - 1) = 33 \\
33 - 10 = 23 \
```

</details>

## Main finished but wrong (24 of 500)

**heldout-n5-00003**: numbers [92, 35, 11, 55, 80], target 72

- base: cut off at 512 tokens, no answer
- main: `92 - 55 + 35 + (80 / 11) * (11 / 80)` — wrong (467 tokens)
- control: cut off at 512 tokens, no answer

<details><summary>Main adapter's full answer</summary>

```text
To create an equation that equals 72 using the numbers 92, 35, 11, 55, and 80 exactly once, we can try the following:

Start by subtracting 55 from 92:  
92 - 55 = 37  

Now, add 35 to that result:  
37 + 35 = 72  

We have used 92, 55, and 35. The remaining numbers are 11 and 80. We need to use them in a way that doesn't affect the result. One way is to subtract and add them in a way that cancels out:  
(80 - 11) = 69, but that doesn't help.  

Instead, we can use division and multiplication to neutralize their effect. For example:  
(80 / 11) is not an integer, but if we do:  
(80 - 11) = 69, and then 69 - 69 = 0, but that doesn't help.  

Let's try a different approach. Try combining 80 and 11:  
80 - 11 = 69  

Now, try combining 92, 35, and 55:  
92 - 55 = 37  
37 + 35 = 72  

Now, we need to incorporate 80 and 11 in a way that doesn't change the result. We can do this by multiplying and dividing them:  
(80 / 11) * (11 / 80) = 1  

So, the full equation is:  
(92 - 55 + 35) + (80 / 11) * (11 / 80) = 72  

Simplifying:  
(92 - 55 + 35) + 1 = 72  

Final answer:  
<answer>92 - 55 + 35 + (80 / 11) * (11 / 80)</answer>
```

</details>

**heldout-n5-00022**: numbers [44, 29, 40, 73, 8], target 32

- base: `(40 - 8) + (73 - 44 - 29)` — correct (472 tokens)
- main: `(73 - 44) + (40 - 8)` — wrong (321 tokens)
- control: `(73 - 44) + (40 - 29) - 8` — correct (220 tokens)

<details><summary>Main adapter's full answer</summary>

```text
To create an equation that equals 32 using the numbers 44, 29, 40, 73, and 8 exactly once, we can try the following:

Start by subtracting two larger numbers to get a smaller value:

$$
73 - 44 = 29
$$

Now we have 29, 40, and 8 left. We can try:

$$
40 - 8 = 32
$$

Now we have used 73, 44, 40, and 8, and the remaining number is 29. We need to incorporate 29 into the equation without changing the result. Notice that:

$$
29 - 29 = 0
$$

So we can add this to the equation:

$$
(73 - 44) + (40 - 8) + (29 - 29) = 32
$$

Simplifying:

$$
29 + 32 + 0 = 32
$$

This works, but we can simplify it further by recognizing that the 29 - 29 cancels out, so we can write:

$$
(73 - 44) + (40 - 8) = 32
$$

This uses all the numbers exactly once and equals 32.

<answer>(73 - 44) + (40 - 8) = 32</answer>
```

</details>
