#import "@preview/ctheorems:1.1.3": *

#show: thmrules.with(qed-symbol: $square$)
#set page(height: auto, margin: 1.5cm)

#let task = thmbox("theorem", "Задание", fill: rgb("#eeffee")).with(numbering: none)
#let solution = thmproof("proof", "Решение")

#align(center, text[= Task02 Максим Исаев ITMO])
#align(center, text[== Теоретическое задание: параллелизуемость/code divergence/memory coalesced access])

#task("1")[
  Пусть на вход дан сигнал `x[n]`, а на выход нужно дать два сигнала `y1[n]` и `y2[n]`:
  ```c
   y1[n] = x[n - 1] + x[n] + x[n + 1]
   y2[n] = y2[n - 2] + y2[n - 1] + x[n]
  ```
  Какой из двух сигналов будет проще и быстрее реализовать в модели массового параллелизма на GPU и почему?
]
#solution()[
  #import "@preview/fletcher:0.5.8" as fletcher: diagram, edge, node

  При реализации `y1` в модели массового параллелизма легче получить значительное ускорение.
  
  Это связано с тем, что элемент `y1[i]` не имеет зависимостей по данным с другими элементами `y1`, таким образом вычисления максимально независимы друг от друга.
  #let node_width = 4.5em
  #figure(
    diagram(
      node-stroke: luma(80%),
      spacing: (0em, 3em),

      node((-1, 0), [`...`], stroke: none),
      node((0, 0), [`x[i]`], name: <x0>, width: node_width),
      node((1, 0), [`x[i+1]`], name: <x1>, width: node_width),
      node((2, 0), [`x[i+2]`], name: <x2>, width: node_width),
      node((3, 0), [`...`], stroke: none),

      node((-1, 1), [`...`], stroke: none),
      node((0, 1), [`y2[i]`], name: <y0>, width: node_width),
      node((1, 1), [`y2[i+1]`], name: <y1>, width: node_width),
      node((2, 1), [`y2[i+2]`], name: <y2>, width: node_width),
      node((3, 1), [`...`], stroke: none),

      edge(<y1>, <x0>, "-|>"),
      edge(<y1>, <x1>, "-|>"),
      edge(<y1>, <x2>, "-|>"),
    ),
    caption: [Зависимость по данным в вычислении `y1[i]`],
  )


  В то время как `y2` имеет зависимости по данным от `y2` (а именно предыдущих двух элементов `y2[i-1]` и `y2[i-2]`).

  Поэтому мы не может начать вычислять `y2[i]` пока не вычислим `y2[i-1]` и `y2[i-2]`, а их в свою очередь нельзя вычислить, пока не будут вычислены `y2[i-2]` и `y2[i-3]` и так далее.

  #figure(
    diagram(
      node-stroke: luma(80%),
      spacing: (0em, 3em),

      node((-1, 0), [`...`], stroke: none),
      node((0, 0), [`x[i]`], name: <x0>, width: node_width),
      node((1, 0), [`x[i+1]`], name: <x1>, width: node_width),
      node((2, 0), [`x[i+2]`], name: <x2>, width: node_width),
      node((3, 0), [`...`], stroke: none),

      node((-1, 1), [`...`], stroke: none),
      node((0, 1), [`y2[i]`], name: <y0>, width: node_width),
      node((1, 1), [`y2[i+1]`], name: <y1>, width: node_width),
      node((2, 1), [`y2[i+2]`], name: <y2>, width: node_width),
      node((3, 1), [`...`], stroke: none),

      edge(<y2>, <x2>, "-|>"),
      edge(<y2>, <y1>, "-|>", bend: -40deg),
      edge(<y2>, <y0>, "-|>", bend: -40deg),
    ),
    caption: [Зависимость по данным в вычислении `y2[i]`],
  )

  *Ответ*: `y1` проще.
]
#pagebreak()

#task("2")[
  Предположим что размер warp/wavefront равен 32 и рабочая группа делится на warp/wavefront-ы таким образом что внутри warp/wavefront номер *WorkItem* по оси `x` меняется чаще всего, затем по оси `y` и затем по оси `z`.

  Напоминание: инструкция исполняется (пусть и отмаскированно) в каждом потоке warp/wavefront если хотя бы один поток выполняет эту инструкцию неотмаскированно. Если не все потоки выполняют эту инструкцию неотмаскированно - происходит т.н. _code divergence_.

  Пусть размер рабочей группы `(32, 32, 1)`
  ```c
  int idx = get_local_id(1) + get_local_size(1) * get_local_id(0);
  if (idx % 32 < 16)
      foo();
  else
      bar();
  ```
  Произойдет ли _code divergence_? Почему?
]
#solution()[
  Разобьем _WorkGroup_ на _warp_'ы.

  #let color = (red, orange, yellow, green, blue, purple)
  #let cols = 32
  #figure(
    table(
      align: center,
      columns: cols + 2,

      table.cell(stroke: none)[],
      table.cell(colspan: cols + 1, stroke: none)[`x`],
      table.cell(rowspan: 8, stroke: none, align: horizon + left)[`y`],
      [],
      ..for value in range(cols) {
        ([#value],)
      },
      ..for i in range(3) {
        ([#i],)
        for value in range(cols) {
          (table.cell(fill: color.at(i))[],)
        }
      },
      ..(table.cell(stroke: none)[...],) * (cols + 1),
      ..for i in range(-3, 0) {
        ([#(32 + i)],)
        for value in range(cols) {
          (table.cell(fill: color.at(i))[],)
        }
      },
    ),
    caption: [_WorkGroup_ (32, 32, 1) разбитый на _warp_'ы],
  )<warp-layout>

  Итого в `j`-ом _warp_'е множество индексов будет выглядеть так:
  - `I(j) = {(i, j) for in 0..32}`.

  Построим множество вычесленных индексов `idx`:
  - `Idx(j) = {j + 32 * i for (i, j) in I(j)}`.

  Применим к каждому элементу `idx` предикат `\idx -> idx % 32 < 16`:
  - `Cond(j) = {idx % 32 < 16 for idx in Idx(j)}`
  - `Cond(j) = {(j + 32 * i) % 32 < 16 for (i, j) in I(j)}`
  - `Cond(j) = {(j + 32 * i) % 32 < 16 for i in 0..32}`
  - `(j + 32 * i) % 32` = `j % 32 + 32 * i % 32` = `j % 32 + 0` = `j`
  - `Cond(j) = {j < 16 for _ in 0..32} = {j < 16}`

  Итого: для каждого _warp_'а `j` значение предиката константно и не зависит от `i`. Это означает что внутри каждого _warp_'а все _WorkItem_'ы либо выполняют `foo()`, либо `bar()`. 
  
  *Ответ*: _code divergence_ не происходит.
]
#pagebreak()

#task("3")[
  Как и в прошлом задании предположим что размер warp/wavefront равен 32 и рабочая группа делится на warp/wavefront-ы таким образом что внутри warp/wavefront номер WorkItem по оси x меняется чаще всего, затем по оси y и затем по оси z.

  Пусть размер рабочей группы (32, 32, 1). Пусть data - указатель на массив float-данных в глобальной видеопамяти идеально выравненный (выравнен по 128 байтам, т.е. data % 128 == 0). И пусть размер кеш линии - 128 байт.

  (a)
  ```
  data[get_local_id(0) + get_local_size(0) * get_local_id(1)] = 1.0f;
  ```
  Будет ли данное обращение к памяти coalesced? Сколько кеш линий записей произойдет в одной рабочей группе?

  (b)

  ```
  data[get_local_id(1) + get_local_size(1) * get_local_id(0)] = 1.0f;
  ```
  Будет ли данное обращение к памяти coalesced? Сколько кеш линий записей произойдет в одной рабочей группе?

  (c)
  ```
  data[1 + get_local_id(0) + get_local_size(0) * get_local_id(1)] = 1.0f;
  ```
  Будет ли данное обращение к памяти coalesced? Сколько кеш линий записей произойдет в одной рабочей группе?
]
#solution()[
  Заметим, что одна кэш-линия вмещает `128 байт / 4 байта = 32` `float`'ов.

  Разбиение на _warp_'ы аналогично предыдущему заданию. (см. #ref(<warp-layout>))
  - `I(j) = {(i, j) for in 0..32}`

  #enum(
    numbering: "(a).",
    enum.item[
      Построим множество индексов по которым произойдет запись:
      - `Idx(j) = {i + 32 * j for (i, j) in I(j)}`
      - `Idx(j) = {(32 * j), (32 * j) + 1, ... , (32 * j) + 31}`.
      Выглядеть это будет так (кэш-линии обозначены цветом, `x` - элемент был записан):
      #figure(
        table(
          align: center,
          columns: (3em,) * 9,
          table.cell(stroke: none)[32j-1],
          table.cell(stroke: none)[32j],
          table.cell(stroke: none)[32j+1],
          table.cell(stroke: none, rowspan: 2, align: horizon)[...],
          table.cell(stroke: none)[32j+i],
          table.cell(stroke: none, rowspan: 2, align: horizon)[...],
          table.cell(stroke: none)[32j+30],
          table.cell(stroke: none)[32j+31],
          table.cell(stroke: none)[32j+32],
          table.cell(fill: red)[],
          ..(table.cell(fill: orange)[`x`],) * 5,
          table.cell(fill: yellow)[],
        ),
        caption: [Записи в массив `data` _warp_'ом под номер `j`.],
      )

      Итого: один _warp_ запишет данные в одну кэш-линию, т.к будет писать 32 последовательных индекса.

    ],
    enum.item[
      Построим множество индексов по которым произойдет запись:
      - `Idx(j) = {j + 32 * i for (i, j) in I(j)}`
      - `Idx(j) = {j, j + 32, ... , j + 32 * 31}`.
      Выглядеть это будет так:
      #figure(
        table(
          align: center,
          columns: (3em,) * 11,
          table.cell(stroke: none)[j-1],
          table.cell(stroke: none)[j],
          table.cell(stroke: none)[j+1],
          table.cell(stroke: none, rowspan: 2, align: horizon)[...],
          table.cell(stroke: none)[j+31],
          table.cell(stroke: none)[j+32],
          table.cell(stroke: none)[j+33],
          table.cell(stroke: none, rowspan: 2, align: horizon)[...],
          table.cell(stroke: none)[j+991],
          table.cell(stroke: none)[j+992],
          table.cell(stroke: none)[j+993],
          table.cell(fill: red)[],
          table.cell(fill: orange)[`x`],
          table.cell(fill: orange)[],
          table.cell(fill: orange)[],
          table.cell(fill: yellow)[`x`],
          table.cell(fill: yellow)[],
          table.cell(fill: blue)[],
          table.cell(fill: purple)[`x`],
          table.cell(fill: purple)[],
        ),
        caption: [Записи в массив `data` _warp_'ом под номер `j`],
      )

      Итого: один _warp_ запишет данные в 32 разные кэш-линии, т.к будет писать в каждый 32 индекс.
    ],
    enum.item[
      Построим множество индексов по которым произойдет запись:
      - `Idx(j) = {1 + i + 32 * j for (i, j) in I(j)}`
      - `Idx(j) = {(32 * j) + 1, (32 * j) + 2, ... , (32 * j) + 32}`.
      Выглядеть это будет так:
      #figure(
        table(
          align: center,
          columns: (3em,) * 10,
          table.cell(stroke: none)[32j-1],
          table.cell(stroke: none)[32j],
          table.cell(stroke: none)[32j+1],
          table.cell(stroke: none, rowspan: 2, align: horizon)[...],
          table.cell(stroke: none)[32j+i],
          table.cell(stroke: none, rowspan: 2, align: horizon)[...],
          table.cell(stroke: none)[32j+30],
          table.cell(stroke: none)[32j+31],
          table.cell(stroke: none)[32j+32],
          table.cell(stroke: none)[32j+33],
          table.cell(fill: red)[],
          table.cell(fill: orange)[],
          ..(table.cell(fill: orange)[`x`],) * 4,
          table.cell(fill: yellow)[`x`],
          table.cell(fill: yellow)[],
        ),
        caption: [Записи в массив `data` _warp_'ом под номер `j`.],
      )

      Итого: один _warp_ запишет данные в 2 кэш-линии, почти все записи придутся на одну кэш-линию, кроме последней: она попадает уже в следующую.
    ],
  )

  *Ответ*:
  #enum(
    numbering: "(a).",
    enum.item[Да, coalesced. 32 кэш-линии на всю рабочую группу.],
    enum.item[Нет, не coalesced. 1024 кэш-линии на всю рабочую группу.],
    enum.item[Будет почти coalesced. 64 кэш-линии на всю рабочую группу.],
  )
]
