20241219：利用SAM2.1，进行车底图像对比，实现异物检测。

20241219：本来想看看现在的对比算法能否在其它领域应用，如果有的话，就能发论文了。但是没发现在其它领域的应用。所以放弃发论文和找其它领域应用的想法吧。

20241219：探索车底3D成像的可能性。
- 首先研究了激光雷达。发现FOV只有90度，一个雷达只能覆盖10cm的车底范围，根本不可行。深度相机也是这个问题。
- [有希望的元器件](https://item.taobao.com/item.htm?abbucket=1&id=649893353771&ns=1&pisk=gouKFBNkbiVQZ6zD-etgEIxTjSAgp4heKvlfr82hVAHtNRbkTyxzyYetUJq3RJb8ybHrEYhyT7wSFYeoxhYmTXzzPKvJoEcFBoJ1_YEQAO_s_S77NhqBKXalTKvDoUKC1zppnvvFtxW_MSa7OkN7CCFuwTsIFu_s1RPzN9N5RCh_QR1CduN7CRN8waa7N8T11WVPRMwCROa_Q7a7OwMS1xFHvRCQ6ank2KzVmQYGDi3T9keRqR7-S270x-Z8Ca9jmWB4hXwOPaenxxhxGYpduWg89cg7YhJK9DnIK5kJkdUrQ0M7wATPC5uKimqKvpQ0UonqZ7HpJwanfcuU-VvAgkH8Bog8XORaSbG_8JlMkIk-iv38mvY6OYz_goi-Kd77KmhsBo0GIZ4rQ0M7wATPC5uKimqKvpQ0UusPxq0v1vB0H7jB6CIP4kNwyTfBCMsllxPTnBFd4gr7_5eD6CIP4kNa6-AdegSzV55..&priceTId=2147bfab17345732729222758e31a3&skuId=4682536364002&spm=a21n57.1.item.29.65ed523cgGDp0q&utparam=%7B%22aplus_abtest%22%3A%226ec2eedc356a6d3d2e04ca60b89c0c32%22%7D&xxc=taobaoSearch)
  - 型号：TOF050C
  - 测量距离：50cm
  - 测距芯片：VL6180
  - 尺寸：2cm x 1.1cm
  - FOV: 25度
  - 电流：40mA
  - 电压：3~5V（DC）