# bookings.parquet schema

Rows: 3,278  (subsample of 120k real bookings)

| column | dtype | meaning |
|---|---|---|
| HOTEL_NAME | str | hotel name as booked |
| DIM_HOTEL_CITY | str | city assigned at booking time (GT) |
| HOTEL_STARS_RATING | int | star rating |
| SUM(BASE_PRICE) | float | aggregate base price |
| SUM(TOTAL_ROOM_NIGHTS) | int | total nights booked |

Source of truth for `ground_truth/gt.json` was built by:
`df.groupby('HOTEL_NAME')['DIM_HOTEL_CITY'].unique()`.
