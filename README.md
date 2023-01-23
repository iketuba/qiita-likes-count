# qiitaのいいね数を予測するアプリ
技術記事を書く人はより多くのいいね数を稼ぎたいと考えており、そのような方にとって有用なサービスを作りたいと考えました。
記事のいいね数の予測値が少ない場合は、本文の内容をより充実させたり、タグの内容を変えたりしてから投稿することで、より多くのいいね数を狙うことができます。

## 担当役割
教師データ収集、予測モデル構築を経て、最後にアプリを実装しました。私は予測モデルの構築やアプリ化で大きく貢献しました。予測モデルの構築では、まず記事のタイトルや内容、タグといった自然言語のデータをMecabというライブラリを用いて分かち書きしました。その後、scikit-learnのTF-IDFを用いて自然言語のデータを数値に変換し、さらにSVDを用いて次元削減しました。以上で全てのデータを数値に変換できたため、最後にLightGBMを使って予測を行いました。また、アプリ化では入力する箇所の実装やモデルを読み込んで予測値を算出する部分の実装を行いました。

## 難しかった点
scikit-learnを使ってTF-IDFの処理を行うと、疎行列で保存されメモリを節約することができます。このデータをpandasのデータフレームに変換すると、メモリの量が足りずエラーが発生してしまいます。そこで、SVDという処理を行い次元削減することで、データ量を抑えています。

## チーム開発で意識した点
メンバーの人が今どのような作業をしているか分かりやすくするために、GitHubでIssueを立ててから作業をするようにしました。さらに、Issueごとに新しいブランチを作ってそこで作業し、開発用のブランチにマージしたい場合はプルリクエストを送ることで、効率的に作業を進めました。
