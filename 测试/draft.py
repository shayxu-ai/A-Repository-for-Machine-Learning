from typing import List


class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid) -> int:
        # 递归
        # 递归的深度
        # 超出时间限制，优化
        # 其实是一道dp题

        if not obstacleGrid or obstacleGrid[0][0] == 1:
            return 0

        len_row = len(obstacleGrid)
        len_col = len(obstacleGrid[0])

        dp = [[0] * len_col for _ in range(len_row)]

        for i in range(len_row):
            for j in range(len_col):
                if i == 0 and j == 0:
                    dp[0][0] = 1
                    continue
                dp[i][j] = 0
                if obstacleGrid[i][j] == 1:
                    continue
                else:
                    if i - 1 >= 0:
                        dp[i][j] += dp[i - 1][j]
                    if j - 1 >= 0:
                        dp[i][j] += dp[i][j - 1]
                print(i, j, dp[i][j])
        return dp[-1][-1]


if __name__ == '__main__':
    s = Solution()
    ret = s.uniquePathsWithObstacles([[0,0,0],[0,1,0],[0,0,0]])
    print(ret)