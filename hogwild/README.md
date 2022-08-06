This version implements [HOGWILD!](https://arxiv.org/pdf/1106.5730.pdf), batching has been removed as the Zodiac dataset did not benefit from batching and the removal made parallelising for hogwild easier. Higher entropy random functions are needed for better training results so `FAST_PREDICTABLE_MODE` was commented out but with no loss in performance thanks to the threading.

This is experimental, and HOGWILD does work pretty well but it does nto yeild more than a 3x performance gain in the best case senarios.
