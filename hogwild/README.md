This version implements [HOGWILD!](https://arxiv.org/pdf/1106.5730.pdf), batching has been removed as the Zodiac dataset did not benefit from batching and the removal made parallelising for hogwild easier. Higher entropy random functions are needed for better training results so `FAST_PREDICTABLE_MODE` was commented out but with no loss in performance thanks to the threading.

This is experimental, and HOGWILD does work pretty well but it does not yeild more than a 3x performance gain in the best case senarios in my tests.

Performance gains tests:
- 0.5x on a 3995WX Zen2 Processor.
- 3x gains on a Zen+ 2700x.
- 0.15x gains on an i7-1165G7.

It is apparent that the gains do not necessarily scale as the raw power of the processor scales, something more intelligent needs to be done to take advantage of the resources available in a 3995WX that has 200 MB/s RAM bandwidth and 128 threads when a 2700x with ~30 MB/s RAM bandwidth and 16 threads is out performing it at a brutally crude multi-threaded application.
