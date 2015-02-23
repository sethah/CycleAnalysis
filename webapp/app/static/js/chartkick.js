Skip to content
Sign up Sign in
Explore
Features
Enterprise
Blog

This repository
Star 296 Fork 22 PUBLICmher/chartkick.py
 branch: master  chartkick.py / chartkick / js / chartkick.js 
Mher Movsisyan mher 7 months ago Updates chartkick.js to 1.1.0
1 contributor
 file  662 lines (578 sloc)  17.55 kb  Open EditRawBlameHistory Delete
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
73
74
75
76
77
78
79
80
81
82
83
84
85
86
87
88
89
90
91
92
93
94
95
96
97
98
99
100
101
102
103
104
105
106
107
108
109
110
111
112
113
114
115
116
117
118
119
120
121
122
123
124
125
126
127
128
129
130
131
132
133
134
135
136
137
138
139
140
141
142
143
144
145
146
147
148
149
150
151
152
153
154
155
156
157
158
159
160
161
162
163
164
165
166
167
168
169
170
171
172
173
174
175
176
177
178
179
180
181
182
183
184
185
186
187
188
189
190
191
192
193
194
195
196
197
198
199
200
201
202
203
204
205
206
207
208
209
210
211
212
213
214
215
216
217
218
219
220
221
222
223
224
225
226
227
228
229
230
231
232
233
234
235
236
237
238
239
240
241
242
243
244
245
246
247
248
249
250
251
252
253
254
255
256
257
258
259
260
261
262
263
264
265
266
267
268
269
270
271
272
273
274
275
276
277
278
279
280
281
282
283
284
285
286
287
288
289
290
291
292
293
294
295
296
297
298
299
300
301
302
303
304
305
306
307
308
309
310
311
312
313
314
315
316
317
318
319
320
321
322
323
324
325
326
327
328
329
330
331
332
333
334
335
336
337
338
339
340
341
342
343
344
345
346
347
348
349
350
351
352
353
354
355
356
357
358
359
360
361
362
363
364
365
366
367
368
369
370
371
372
373
374
375
376
377
378
379
380
381
382
383
384
385
386
387
388
389
390
391
392
393
394
395
396
397
398
399
400
401
402
403
404
405
406
407
408
409
410
411
412
413
414
415
416
417
418
419
420
421
422
423
424
425
426
427
428
429
430
431
432
433
434
435
436
437
438
439
440
441
442
443
444
445
446
447
448
449
450
451
452
453
454
455
456
457
458
459
460
461
462
463
464
465
466
467
468
469
470
471
472
473
474
475
476
477
478
479
480
481
482
483
484
485
486
487
488
489
490
491
492
493
494
495
496
497
498
499
500
501
502
503
504
505
506
507
508
509
510
511
512
513
514
515
516
517
518
519
520
521
522
523
524
525
526
527
528
529
530
531
532
533
534
535
536
537
538
539
540
541
542
543
544
545
546
547
548
549
550
551
552
553
554
555
556
557
558
559
560
561
562
563
564
565
566
567
568
569
570
571
572
573
574
575
576
577
578
579
580
581
582
583
584
585
586
587
588
589
590
591
592
593
594
595
596
597
598
599
600
601
602
603
604
605
606
607
608
609
610
611
612
613
614
615
616
617
618
619
620
621
622
623
624
625
626
627
628
629
630
631
632
633
634
635
636
637
638
639
640
641
642
643
644
645
646
647
648
649
650
651
652
653
654
655
656
657
658
659
660
661
/*
 * Chartkick.js
 * Create beautiful Javascript charts with minimal code
 * https://github.com/ankane/chartkick.js
 * v1.1.0
 * MIT License
 */

/*jslint browser: true, indent: 2, plusplus: true, vars: true */
/*global google, Highcharts, $*/

(function () {
  'use strict';

  var Chartkick, ISO8601_PATTERN, DECIMAL_SEPARATOR, defaultOptions, hideLegend,
    setMin, setMax, jsOptions, loaded, waitForLoaded, setBarMin, setBarMax, createDataTable, resize;

  // only functions that need defined specific to charting library
  var renderLineChart, renderPieChart, renderColumnChart, renderBarChart, renderAreaChart;

  // helpers

  function isArray(variable) {
    return Object.prototype.toString.call(variable) === "[object Array]";
  }

  function isFunction(variable) {
    return variable instanceof Function;
  }

  function isPlainObject(variable) {
    return !isFunction(variable) && variable instanceof Object;
  }

  // https://github.com/madrobby/zepto/blob/master/src/zepto.js
  function extend(target, source) {
    var key;
    for (key in source) {
      if (isPlainObject(source[key]) || isArray(source[key])) {
        if (isPlainObject(source[key]) && !isPlainObject(target[key])) {
          target[key] = {};
        }
        if (isArray(source[key]) && !isArray(target[key])) {
          target[key] = [];
        }
        extend(target[key], source[key]);
      } else if (source[key] !== undefined) {
        target[key] = source[key];
      }
    }
  }

  function merge(obj1, obj2) {
    var target = {};
    extend(target, obj1);
    extend(target, obj2);
    return target;
  }

  // https://github.com/Do/iso8601.js
  ISO8601_PATTERN = /(\d\d\d\d)(\-)?(\d\d)(\-)?(\d\d)(T)?(\d\d)(:)?(\d\d)?(:)?(\d\d)?([\.,]\d+)?($|Z|([\+\-])(\d\d)(:)?(\d\d)?)/i;
  DECIMAL_SEPARATOR = String(1.5).charAt(1);

  function parseISO8601(input) {
    var day, hour, matches, milliseconds, minutes, month, offset, result, seconds, type, year;
    type = Object.prototype.toString.call(input);
    if (type === '[object Date]') {
      return input;
    }
    if (type !== '[object String]') {
      return;
    }
    if (matches = input.match(ISO8601_PATTERN)) {
      year = parseInt(matches[1], 10);
      month = parseInt(matches[3], 10) - 1;
      day = parseInt(matches[5], 10);
      hour = parseInt(matches[7], 10);
      minutes = matches[9] ? parseInt(matches[9], 10) : 0;
      seconds = matches[11] ? parseInt(matches[11], 10) : 0;
      milliseconds = matches[12] ? parseFloat(DECIMAL_SEPARATOR + matches[12].slice(1)) * 1000 : 0;
      result = Date.UTC(year, month, day, hour, minutes, seconds, milliseconds);
      if (matches[13] && matches[14]) {
        offset = matches[15] * 60;
        if (matches[17]) {
          offset += parseInt(matches[17], 10);
        }
        offset *= matches[14] === '-' ? -1 : 1;
        result -= offset * 60 * 1000;
      }
      return new Date(result);
    }
  }
  // end iso8601.js

  function negativeValues(series) {
    var i, j, data;
    for (i = 0; i < series.length; i++) {
      data = series[i].data;
      for (j = 0; j < data.length; j++) {
        if (data[j][1] < 0) {
          return true;
        }
      }
    }
    return false;
  }

  function jsOptionsFunc(defaultOptions, hideLegend, setMin, setMax) {
    return function (series, opts, chartOptions) {
      var options = merge({}, defaultOptions);
      options = merge(options, chartOptions || {});

      // hide legend
      // this is *not* an external option!
      if (opts.hideLegend) {
        hideLegend(options);
      }

      // min
      if ("min" in opts) {
        setMin(options, opts.min);
      } else if (!negativeValues(series)) {
        setMin(options, 0);
      }

      // max
      if ("max" in opts) {
        setMax(options, opts.max);
      }

      // merge library last
      options = merge(options, opts.library || {});

      return options;
    };
  }

  function setText(element, text) {
    if (document.body.innerText) {
      element.innerText = text;
    } else {
      element.textContent = text;
    }
  }

  function chartError(element, message) {
    setText(element, "Error Loading Chart: " + message);
    element.style.color = "#ff0000";
  }

  function getJSON(element, url, success) {
    $.ajax({
      dataType: "json",
      url: url,
      success: success,
      error: function (jqXHR, textStatus, errorThrown) {
        var message = (typeof errorThrown === "string") ? errorThrown : errorThrown.message;
        chartError(element, message);
      }
    });
  }

  function errorCatcher(element, data, opts, callback) {
    try {
      callback(element, data, opts);
    } catch (err) {
      chartError(element, err.message);
      throw err;
    }
  }

  function fetchDataSource(element, dataSource, opts, callback) {
    if (typeof dataSource === "string") {
      getJSON(element, dataSource, function (data, textStatus, jqXHR) {
        errorCatcher(element, data, opts, callback);
      });
    } else {
      errorCatcher(element, dataSource, opts, callback);
    }
  }

  // type conversions

  function toStr(n) {
    return "" + n;
  }

  function toFloat(n) {
    return parseFloat(n);
  }

  function toDate(n) {
    if (typeof n !== "object") {
      if (typeof n === "number") {
        n = new Date(n * 1000); // ms
      } else { // str
        // try our best to get the str into iso8601
        // TODO be smarter about this
        var str = n.replace(/ /, "T").replace(" ", "").replace("UTC", "Z");
        n = parseISO8601(str) || new Date(n);
      }
    }
    return n;
  }

  function toArr(n) {
    if (!isArray(n)) {
      var arr = [], i;
      for (i in n) {
        if (n.hasOwnProperty(i)) {
          arr.push([i, n[i]]);
        }
      }
      n = arr;
    }
    return n;
  }

  function sortByTime(a, b) {
    return a[0].getTime() - b[0].getTime();
  }

  if ("Highcharts" in window) {

    defaultOptions = {
      chart: {},
      xAxis: {
        labels: {
          style: {
            fontSize: "12px"
          }
        }
      },
      yAxis: {
        title: {
          text: null
        },
        labels: {
          style: {
            fontSize: "12px"
          }
        }
      },
      title: {
        text: null
      },
      credits: {
        enabled: false
      },
      legend: {
        borderWidth: 0
      },
      tooltip: {
        style: {
          fontSize: "12px"
        }
      },
      plotOptions: {
        areaspline: {},
        series: {
          marker: {}
        }
      }
    };

    hideLegend = function (options) {
      options.legend.enabled = false;
    };

    setMin = function (options, min) {
      options.yAxis.min = min;
    };

    setMax = function (options, max) {
      options.yAxis.max = max;
    };

    jsOptions = jsOptionsFunc(defaultOptions, hideLegend, setMin, setMax);

    renderLineChart = function (element, series, opts, chartType) {
      chartType = chartType || "spline";
      var chartOptions = {};
      if (chartType === "areaspline") {
        chartOptions = {
          plotOptions: {
            areaspline: {
              stacking: "normal"
            },
            series: {
              marker: {
                enabled: false
              }
            }
          }
        };
      }
      var options = jsOptions(series, opts, chartOptions), data, i, j;
      options.xAxis.type = "datetime";
      options.chart.type = chartType;
      options.chart.renderTo = element.id;

      for (i = 0; i < series.length; i++) {
        data = series[i].data;
        for (j = 0; j < data.length; j++) {
          data[j][0] = data[j][0].getTime();
        }
        series[i].marker = {symbol: "circle"};
      }
      options.series = series;
      new Highcharts.Chart(options);
    };

    renderPieChart = function (element, series, opts) {
      var options = merge(defaultOptions, opts.library || {});
      options.chart.renderTo = element.id;
      options.series = [{
        type: "pie",
        name: "Value",
        data: series
      }];
      new Highcharts.Chart(options);
    };

    renderColumnChart = function (element, series, opts, chartType) {
      chartType = chartType || "column";
      var options = jsOptions(series, opts), i, j, s, d, rows = [];
      options.chart.type = chartType;
      options.chart.renderTo = element.id;

      for (i = 0; i < series.length; i++) {
        s = series[i];

        for (j = 0; j < s.data.length; j++) {
          d = s.data[j];
          if (!rows[d[0]]) {
            rows[d[0]] = new Array(series.length);
          }
          rows[d[0]][i] = d[1];
        }
      }

      var categories = [];
      for (i in rows) {
        if (rows.hasOwnProperty(i)) {
          categories.push(i);
        }
      }
      options.xAxis.categories = categories;

      var newSeries = [];
      for (i = 0; i < series.length; i++) {
        d = [];
        for (j = 0; j < categories.length; j++) {
          d.push(rows[categories[j]][i] || 0);
        }

        newSeries.push({
          name: series[i].name,
          data: d
        });
      }
      options.series = newSeries;

      new Highcharts.Chart(options);
    };

    renderBarChart = function (element, series, opts) {
      renderColumnChart(element, series, opts, "bar");
    };

    renderAreaChart = function (element, series, opts) {
      renderLineChart(element, series, opts, "areaspline");
    };
  } else if ("google" in window) { // Google charts
    // load from google
    loaded = false;
    google.setOnLoadCallback(function () {
      loaded = true;
    });
    google.load("visualization", "1.0", {"packages": ["corechart"]});

    waitForLoaded = function (callback) {
      google.setOnLoadCallback(callback); // always do this to prevent race conditions (watch out for other issues due to this)
      if (loaded) {
        callback();
      }
    };

    // Set chart options
    defaultOptions = {
      chartArea: {},
      fontName: "'Lucida Grande', 'Lucida Sans Unicode', Verdana, Arial, Helvetica, sans-serif",
      pointSize: 6,
      legend: {
        textStyle: {
          fontSize: 12,
          color: "#444"
        },
        alignment: "center",
        position: "right"
      },
      curveType: "function",
      hAxis: {
        textStyle: {
          color: "#666",
          fontSize: 12
        },
        gridlines: {
          color: "transparent"
        },
        baselineColor: "#ccc",
        viewWindow: {}
      },
      vAxis: {
        textStyle: {
          color: "#666",
          fontSize: 12
        },
        baselineColor: "#ccc",
        viewWindow: {}
      },
      tooltip: {
        textStyle: {
          color: "#666",
          fontSize: 12
        }
      }
    };

    hideLegend = function (options) {
      options.legend.position = "none";
    };

    setMin = function (options, min) {
      options.vAxis.viewWindow.min = min;
    };

    setMax = function (options, max) {
      options.vAxis.viewWindow.max = max;
    };

    setBarMin = function (options, min) {
      options.hAxis.viewWindow.min = min;
    };

    setBarMax = function (options, max) {
      options.hAxis.viewWindow.max = max;
    };

    jsOptions = jsOptionsFunc(defaultOptions, hideLegend, setMin, setMax);

    // cant use object as key
    createDataTable = function (series, columnType) {
      var data = new google.visualization.DataTable();
      data.addColumn(columnType, "");

      var i, j, s, d, key, rows = [];
      for (i = 0; i < series.length; i++) {
        s = series[i];
        data.addColumn("number", s.name);

        for (j = 0; j < s.data.length; j++) {
          d = s.data[j];
          key = (columnType === "datetime") ? d[0].getTime() : d[0];
          if (!rows[key]) {
            rows[key] = new Array(series.length);
          }
          rows[key][i] = toFloat(d[1]);
        }
      }

      var rows2 = [];
      for (i in rows) {
        if (rows.hasOwnProperty(i)) {
          rows2.push([(columnType === "datetime") ? new Date(toFloat(i)) : i].concat(rows[i]));
        }
      }
      if (columnType === "datetime") {
        rows2.sort(sortByTime);
      }
      data.addRows(rows2);

      return data;
    };

    resize = function (callback) {
      if (window.attachEvent) {
        window.attachEvent("onresize", callback);
      } else if (window.addEventListener) {
        window.addEventListener("resize", callback, true);
      }
      callback();
    };

    renderLineChart = function (element, series, opts) {
      waitForLoaded(function () {
        var options = jsOptions(series, opts);
        var data = createDataTable(series, "datetime");
        var chart = new google.visualization.LineChart(element);
        resize(function () {
          chart.draw(data, options);
        });
      });
    };

    renderPieChart = function (element, series, opts) {
      waitForLoaded(function () {
        var chartOptions = {
          chartArea: {
            top: "10%",
            height: "80%"
          }
        };
        var options = merge(merge(defaultOptions, chartOptions), opts.library || {});

        var data = new google.visualization.DataTable();
        data.addColumn("string", "");
        data.addColumn("number", "Value");
        data.addRows(series);

        var chart = new google.visualization.PieChart(element);
        resize(function () {
          chart.draw(data, options);
        });
      });
    };

    renderColumnChart = function (element, series, opts) {
      waitForLoaded(function () {
        var options = jsOptions(series, opts);
        var data = createDataTable(series, "string");
        var chart = new google.visualization.ColumnChart(element);
        resize(function () {
          chart.draw(data, options);
        });
      });
    };

    renderBarChart = function (element, series, opts) {
      waitForLoaded(function () {
        var chartOptions = {
          hAxis: {
            gridlines: {
              color: "#ccc"
            }
          }
        };
        var options = jsOptionsFunc(defaultOptions, hideLegend, setBarMin, setBarMax)(series, opts, chartOptions);
        var data = createDataTable(series, "string");
        var chart = new google.visualization.BarChart(element);
        resize(function () {
          chart.draw(data, options);
        });
      });
    };

    renderAreaChart = function (element, series, opts) {
      waitForLoaded(function () {
        var chartOptions = {
          isStacked: true,
          pointSize: 0,
          areaOpacity: 0.5
        };
        var options = jsOptions(series, opts, chartOptions);
        var data = createDataTable(series, "datetime");
        var chart = new google.visualization.AreaChart(element);
        resize(function () {
          chart.draw(data, options);
        });
      });
    };
  } else { // no chart library installed
    renderLineChart = renderPieChart = renderColumnChart = renderBarChart = renderAreaChart = function () {
      throw new Error("Please install Google Charts or Highcharts");
    };
  }

  // process data

  function processSeries(series, opts, time) {
    var i, j, data, r, key;

    // see if one series or multiple
    if (!isArray(series) || typeof series[0] !== "object" || isArray(series[0])) {
      series = [{name: "Value", data: series}];
      opts.hideLegend = true;
    } else {
      opts.hideLegend = false;
    }

    // right format
    for (i = 0; i < series.length; i++) {
      data = toArr(series[i].data);
      r = [];
      for (j = 0; j < data.length; j++) {
        key = data[j][0];
        key = time ? toDate(key) : toStr(key);
        r.push([key, toFloat(data[j][1])]);
      }
      if (time) {
        r.sort(sortByTime);
      }
      series[i].data = r;
    }

    return series;
  }

  function processLineData(element, data, opts) {
    renderLineChart(element, processSeries(data, opts, true), opts);
  }

  function processColumnData(element, data, opts) {
    renderColumnChart(element, processSeries(data, opts, false), opts);
  }

  function processPieData(element, data, opts) {
    var perfectData = toArr(data), i;
    for (i = 0; i < perfectData.length; i++) {
      perfectData[i] = [toStr(perfectData[i][0]), toFloat(perfectData[i][1])];
    }
    renderPieChart(element, perfectData, opts);
  }

  function processBarData(element, data, opts) {
    renderBarChart(element, processSeries(data, opts, false), opts);
  }

  function processAreaData(element, data, opts) {
    renderAreaChart(element, processSeries(data, opts, true), opts);
  }

  function setElement(element, data, opts, callback) {
    if (typeof element === "string") {
      element = document.getElementById(element);
    }
    fetchDataSource(element, data, opts || {}, callback);
  }

  // define classes

  Chartkick = {
    LineChart: function (element, dataSource, opts) {
      setElement(element, dataSource, opts, processLineData);
    },
    PieChart: function (element, dataSource, opts) {
      setElement(element, dataSource, opts, processPieData);
    },
    ColumnChart: function (element, dataSource, opts) {
      setElement(element, dataSource, opts, processColumnData);
    },
    BarChart: function (element, dataSource, opts) {
      setElement(element, dataSource, opts, processBarData);
    },
    AreaChart: function (element, dataSource, opts) {
      setElement(element, dataSource, opts, processAreaData);
    }
  };

  window.Chartkick = Chartkick;
}());
Status API Training Shop Blog About © 2014 GitHub, Inc. Terms Privacy Security Contact