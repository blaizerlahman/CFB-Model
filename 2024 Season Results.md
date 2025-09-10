## 2024 Season Model Results

The overall results of the model are as follows:
- Overall win rate: 47.2%, or 324-363
- Win rate on "good bets" (59.5% win rate): 48.2%, or 150-156
- Win rate on "great bets" (64.5% win rate): 45.7%, or 48-56
- Win rate on "best bets" (69.5% win rate): 53.8%, or 7-6

Obviously, these are not great results. However, a majority of the negative results come from the first half of the season, in which we went 64-83 on "good bets" with similar results in other bet categories.

The second half of the season actually performed much better, with the results being as follows:
- Overall win rate: 49.9%, or 185-186
- Win rate on "good bets": 53.8%, or 86-73
- Win rate on "great bets": 50.9%, or 29-28
- Win rate on "best bets": 57.1%, or 4-3

This can most likely be attributed to the fact that, by week 8, the model's short term trends for each team (their 8 week rolling averages) had now fully been comprised of only stats from the current season's team.
Thus our model now has a better grasp on how the team is currently playing, and has almost fully baked out performances from the previous season. The same can also be said for SP+ ratings, as now they are adjusted to the current teams' performances.

We know now that, when training the models, we were using post-season SP+ ratings to predict mid-season games. This skewed the results of the training by a huge amount and likely had an impact on this season's results. 

Next season we plan on adding more advanced metrics and exploring different ways of evaluating model success.
