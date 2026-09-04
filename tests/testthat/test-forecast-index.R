# The future index must come back in the observed index's own type. Extending
# in numeric space silently returned bare numbers for every tsibble calendar
# type, which breaks the join back to the caller's data and the interval a
# tsibble reports.

panel_with_index <- function(idx) {
  data.frame(
    store = "a",
    when  = idx,
    sales = as.numeric(seq_along(idx)) + sin(seq_along(idx))
  )
}

forecast_index <- function(idx, h = 3L) {
  fc <- forecast(
    zuk_pretrained("stub"), panel_with_index(idx), h = h,
    index = "when", key = "store", target = "sales"
  )
  on.exit(zuk_unload("stub"), add = TRUE)
  fc[["when"]]
}

test_that("a numeric or integer index keeps its type", {
  numeric_out <- forecast_index(as.numeric(1:36))
  expect_type(numeric_out, "double")
  expect_equal(numeric_out, c(37, 38, 39))

  integer_out <- forecast_index(1:36)
  expect_type(integer_out, "integer")
  expect_identical(integer_out, 37:39)
})

test_that("a Date index keeps its class and its observed spacing", {
  daily <- forecast_index(seq(as.Date("2020-01-01"), by = "day", length.out = 36))
  expect_s3_class(daily, "Date")
  expect_equal(daily, as.Date(c("2020-02-06", "2020-02-07", "2020-02-08")))

  weekly <- forecast_index(seq(as.Date("2020-01-06"), by = "week", length.out = 36))
  expect_s3_class(weekly, "Date")
  expect_identical(as.numeric(diff(weekly)), c(7, 7))
})

test_that("a POSIXct index keeps its class and time zone", {
  hourly <- forecast_index(as.POSIXct("2020-01-01", tz = "UTC") + (0:35) * 3600)
  expect_s3_class(hourly, "POSIXct")
  expect_identical(attr(hourly, "tzone"), "UTC")
  # Storage type is seq()'s business; the instants are what matter.
  expect_equal(
    hourly,
    as.POSIXct("2020-01-02 12:00:00", tz = "UTC") + (0:2) * 3600
  )
})

test_that("tsibble calendar indexes are extended in their own periods", {
  skip_if_not_installed("tsibble")

  months <- forecast_index(
    tsibble::yearmonth(seq(as.Date("2020-01-01"), by = "month", length.out = 36))
  )
  expect_s3_class(months, "yearmonth")
  expect_identical(months, tsibble::yearmonth(c("2023 Jan", "2023 Feb", "2023 Mar")))

  quarters <- forecast_index(
    tsibble::yearquarter(seq(as.Date("2020-01-01"), by = "quarter", length.out = 36))
  )
  expect_s3_class(quarters, "yearquarter")
  expect_identical(quarters[[1]], tsibble::yearquarter("2029 Q1"))

  weeks <- forecast_index(
    tsibble::yearweek(seq(as.Date("2020-01-06"), by = "week", length.out = 36))
  )
  expect_s3_class(weeks, "yearweek")
  expect_identical(as.numeric(diff(as.Date(weeks))), c(7, 7))
})

test_that("a fable built from a calendar index reports the right interval", {
  skip_if_not_installed("tsibble")
  skip_if_not_installed("fabletools")

  history <- tsibble::tsibble(
    when  = tsibble::yearmonth(seq(as.Date("2020-01-01"), by = "month", length.out = 36)),
    store = "a",
    sales = as.numeric(1:36),
    index = when, key = store
  )
  model <- zuk_pretrained("stub")
  on.exit(zuk_unload("stub"), add = TRUE)

  fable <- fabletools::as_fable(forecast(model, history, h = 3))
  expect_s3_class(fable[["when"]], "yearmonth")
  expect_identical(format(tsibble::interval(fable)), "1M")
})

test_that("an index type that cannot be extended is refused, not downgraded", {
  model <- zuk_pretrained("stub")
  on.exit(zuk_unload("stub"), add = TRUE)

  character_index <- data.frame(
    when  = letters[1:10],
    sales = as.numeric(1:10)
  )
  expect_error(
    forecast(model, character_index, h = 2, index = "when", target = "sales"),
    class = "zuk_error_capability"
  )
})
