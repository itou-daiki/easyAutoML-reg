import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
import joblib
import base64
import io
import shap
from pycaret.regression import *
from sklearn.tree import plot_tree
from sklearn.inspection import permutation_importance
import matplotlib.font_manager as fm

# フォントファイルのパスを指定
font_path = 'ipaexg.ttf'  # フォントファイルのパスを正確に指定してください

# フォントプロパティを作成
font_prop = fm.FontProperties(fname=font_path)

# フォントをMatplotlibのデフォルトフォントに設定
plt.rcParams['font.family'] = font_prop.get_name()

# ページ設定
st.set_page_config(
    page_title="easyAutoML（回帰）",
    page_icon="📊",
    layout="wide"
)

# セッションステートの初期化
if 'model_configs' not in st.session_state:
    st.session_state.model_configs = {
        'uploaded_data': None,
        'target_variable': None,
        'ignore_features': [],
        'setup_done': False,
        'model_trained': False,
        'current_model': None,
        'final_model': None,
        'feature_importance': None,
        'model_comparison': None,
        'pre_tuned_scores': None,
        'post_tuned_scores': None,
        'X_train': None,
        'features': None
    }

# メインアプリケーション
st.title("easyAutoML（回帰）")
st.caption("Created by Dit-Lab.(Daiki Ito)")

with st.expander("このアプリケーションについて", expanded=False):
    st.markdown("""
    ### 📊 AutoMLアプリケーションの概要
    このアプリケーションは、機械学習モデルの構築と評価を自動化し、詳細な分析結果を提供します。
    """)

    st.markdown("### 🔍 主な機能")
    
    st.markdown("#### 1. データ分析")
    st.markdown("""
    - データの基本統計量の確認
    - 欠損値の分析と自動処理
    - 外れ値の検出と処理オプション
    """)

    st.markdown("#### 2. モデル構築と最適化")
    st.markdown("""
    - 複数の機械学習モデルの自動比較
    - 最適なモデルの選択支援
    - ハイパーパラメータの自動チューニング
    - 交差検証による性能評価
    """)

    st.markdown("#### 3. モデル評価と解釈")
    st.markdown("""
    - 特徴量重要度の分析
        - モデルベースの重要度
        - SHAP値による解釈
    - 予測性能の詳細な評価
        - 残差分析
        - 予測値と実測値の比較
    - クックの距離による影響度分析
    - 決定木モデルの構造可視化（対応モデルのみ）
    """)

    st.markdown("#### 4. 最終モデルの生成")
    st.markdown("""
    - モデルのファイナライズ
    - 最終評価結果の確認
    - トレーニング済みモデルのダウンロード
    """)

    st.markdown("### 💡 使用方法")
    st.markdown("""
    1. CSVまたはExcelファイルをアップロード
    2. 予測したい目的変数を選択
    3. モデルの比較と選択
    4. 選択したモデルのチューニングと評価
    5. 最終モデルの保存
    """)

    st.info("⚠️ 注意: データの前処理（欠損値の処理など）は自動的に行われますが、データの品質が結果に大きく影響します。")

# 1. データアップロード部分
st.markdown("---")
st.header("1. データのアップロード")
st.write("データのアップロードについて")
st.markdown("""
- CSVまたはExcelファイルをアップロードしてください
- 欠損値や異常値は自動的に処理されます
""")

uploaded_file = st.file_uploader(
    "ファイルを選択してください（CSVまたはExcel）",
    type=['csv', 'xlsx']
)

if uploaded_file is not None:
    try:
        # 新しいファイルがアップロードされた場合のセッションリセット
        if 'last_file_name' not in st.session_state or st.session_state.last_file_name != uploaded_file.name:
            st.session_state.model_configs = {
                'uploaded_data': None,
                'target_variable': None,
                'ignore_features': [],
                'setup_done': False,
                'model_trained': False,
                'current_model': None,
                'final_model': None,
                'feature_importance': None,
                'model_comparison': None,
                'pre_tuned_scores': None,
                'post_tuned_scores': None,
                'X_train': None,
                'features': None
            }
            st.session_state.last_file_name = uploaded_file.name

        # データ読み込み
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)
        
        st.session_state.model_configs['uploaded_data'] = data
        
        # データの基本情報表示
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("データ件数", f"{len(data):,}件")
        with col2:
            st.metric("項目数", f"{len(data.columns)}個")
        with col3:
            st.metric("欠損値を含む列", f"{data.isnull().any().sum()}個")
        
        # データプレビュー
        st.subheader("データプレビュー")
        st.dataframe(data.head(), use_container_width=True)
        
        # 基本統計量
        with st.expander("データの基本統計量を表示", expanded=False):
            st.dataframe(data.describe(), use_container_width=True)
        
        # 欠損値の情報
        if data.isnull().any().sum() > 0:
            with st.expander("欠損値の詳細を表示", expanded=False):
                missing_data = pd.DataFrame({
                    '欠損値数': data.isnull().sum(),
                    '欠損率(%)': (data.isnull().sum() / len(data) * 100).round(2)
                }).reset_index()
                missing_data.columns = ['列名', '欠損値数', '欠損率(%)']
                st.dataframe(missing_data[missing_data['欠損値数'] > 0], use_container_width=True)

        st.markdown("---")
        # 2. モデル設定
        st.header("2. モデル設定")
        
        col1, col2 = st.columns(2)
        with col1:
            target_variable = st.selectbox(
                '予測対象（目的変数）の選択',
                options=data.columns,
                help="予測したい項目を選択してください"
            )
            st.session_state.model_configs['target_variable'] = target_variable

        with col2:
            ignore_features = st.multiselect(
                '除外する項目の選択',
                options=[col for col in data.columns if col != target_variable],
                help="モデルの学習に使用しない項目を選択してください"
            )
            st.session_state.model_configs['ignore_features'] = ignore_features

        # データ処理オプション
        col1, col2 = st.columns(2)
        with col1:
            remove_outliers = st.checkbox('外れ値を除去する', value=False)

        # モデル比較の実行
        if st.button('モデルの比較を開始', use_container_width=True):
            try:
                with st.spinner("モデルを比較中..."):
                    # プログレスバーの表示
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # データのセットアップ
                    status_text.text("データの前処理を実行中...")
                    progress_bar.progress(20)
                    
                    # セットアップの実行
                    setup_data = setup(
                        data=data,
                        target=target_variable,
                        ignore_features=ignore_features,
                        remove_outliers=remove_outliers,
                        session_id=123,
                        verbose=False
                    )
                    
                    # 特徴量の保存
                    X_train = get_config('X_train')
                    st.session_state.model_configs['X_train'] = X_train
                    st.session_state.model_configs['features'] = X_train.columns.tolist()
                    
                    # モデルの比較
                    progress_bar.progress(40)
                    status_text.text("モデルを比較中...")
                    
                    models_comparison = compare_models(
                        exclude=['catboost'],
                        fold=5,
                        sort='MAE',
                        n_select=15,
                        verbose=False
                    )
                    
                    # 比較結果を保存
                    comparison_df = pull()
                    st.session_state.model_configs['model_comparison'] = comparison_df.copy()
                    
                    progress_bar.progress(100)
                    status_text.text("モデルの比較が完了しました！")
                    st.success("✅ モデルの比較が完了しました！")

            except Exception as e:
                st.error(f"エラーが発生しました: {str(e)}")
                st.stop()

        # モデルの選択とチューニング
        if 'model_comparison' in st.session_state.model_configs and st.session_state.model_configs['model_comparison'] is not None:
            st.markdown("---")
            st.header("モデルの選択とチューニング")
            
            # モデル比較結果の表示
            st.subheader("モデル比較結果")
            st.dataframe(st.session_state.model_configs['model_comparison'], use_container_width=True)
            with st.expander("モデルの説明", expanded=False):
                st.markdown("""
                **モデルの説明をここに記載**（省略）
                """)

            # 比較結果の説明
            with st.expander("評価指標の説明", expanded=False):
                st.markdown("""
                **評価指標の説明をここに記載**（省略）
                """)
            
            # モデルの選択
            selected_model = st.selectbox(
                'チューニングするモデルを選択',
                options=st.session_state.model_configs['model_comparison'].index,
                help="比較結果から最適なモデルを選択してください"
            )
            
            # ハイパーパラメータの設定
            st.subheader("ハイパーパラメータの設定")
            
            # 交差検証の設定
            cv_folds = st.slider('交差検証のfold数', min_value=2, max_value=20, value=5)

            # モデル固有のハイパーパラメータ設定
            params = {}
            if selected_model in ['rf', 'et']:
                n_estimators = st.slider('決定木の数', min_value=50, max_value=500, value=100, step=50)
                max_depth = st.slider('決定木の最大深さ', min_value=3, max_value=20, value=3)
                min_samples_split = st.slider('分割のための最小サンプル数', min_value=2, max_value=20, value=2)
                min_samples_leaf = st.slider('葉となるための最小サンプル数', min_value=1, max_value=10, value=1)
                params = {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf
                }
            elif selected_model in ['xgboost', 'lightgbm']:
                learning_rate = st.slider('学習率', min_value=0.01, max_value=0.3, value=0.1, step=0.01)
                n_estimators = st.slider('決定木の数', min_value=50, max_value=500, value=100, step=50)
                max_depth = st.slider('決定木の最大深さ', min_value=3, max_value=20, value=3)
                params = {
                    'learning_rate': learning_rate,
                    'n_estimators': n_estimators,
                    'max_depth': max_depth
                }

            # モデルのトレーニングとチューニング
            if st.button('選択したモデルでトレーニング開始', use_container_width=True):
                try:
                    with st.spinner("モデルをトレーニング中..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        # 選択したモデルの作成
                        status_text.text("モデルを作成中...")
                        progress_bar.progress(30)
                        
                        # モデルの作成とスコアの保存
                        base_model = create_model(selected_model, fold=cv_folds, **params)
                        pre_tuned_scores = pull()
                        st.session_state.model_configs['pre_tuned_scores'] = pre_tuned_scores.copy()
                        
                        # チューニング
                        status_text.text("ハイパーパラメータをチューニング中...")
                        progress_bar.progress(60)
                        tuned_model = tune_model(
                            base_model,
                            n_iter=10,
                            fold=cv_folds,
                            optimize='MAE'
                        )
                        
                        # チューニング後のスコアを保存
                        post_tuned_scores = pull()
                        st.session_state.model_configs.update({
                            'current_model': tuned_model,
                            'post_tuned_scores': post_tuned_scores.copy(),
                            'model_trained': True
                        })
                        
                        # チューニング結果の表示
                        st.subheader("チューニング結果")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("チューニング前")
                            st.dataframe(pre_tuned_scores, use_container_width=True)
                        with col2:
                            st.write("チューニング後")
                            st.dataframe(post_tuned_scores, use_container_width=True)
                        
                        progress_bar.progress(100)
                        status_text.text("モデルのトレーニングが完了しました！")
                        st.success("✅ モデルのトレーニングが完了しました！")

                except Exception as e:
                    st.error(f"モデルのトレーニング中にエラーが発生しました: {str(e)}")
                    st.stop()

        # モデル評価
        if st.session_state.model_configs['model_trained']:
            st.markdown("---")
            st.header("3. モデルの評価")
        
            # 1. 特徴量重要度の分析
            st.subheader("3-1. 特徴量重要度")
            try:
                model = st.session_state.model_configs['current_model']
                X_train_transformed = get_config('X_train_transformed')
                feature_names = X_train_transformed.columns
            
                col1, col2 = st.columns(2)
            
                with col1:
                    st.write("モデルベースの特徴量重要度")
                    with st.spinner('特徴量重要度を計算中...'):
                        try:
                            if hasattr(model, 'feature_importances_'):
                                importance_df = pd.DataFrame({
                                    '特徴量': feature_names,
                                    '重要度': model.feature_importances_
                                }).sort_values(by='重要度', ascending=False)
                                # プロット
                                fig, ax = plt.subplots(figsize=(10, 6))
                                sns.barplot(data=importance_df, x='重要度', y='特徴量', ax=ax)
                                st.pyplot(fig)
                                # キャプションを追加
                                st.caption("**モデルが学習した結果、各特徴量が目的変数の予測にどれだけ寄与したかを示しています。重要度が高い特徴量ほど、モデルの予測に大きな影響を与えています。**")
                                # データフレームを表示
                                with st.expander("特徴量重要度データ", expanded=False):
                                    st.dataframe(importance_df, use_container_width=True)
                            else:
                                st.warning("このモデルでは特徴量重要度を直接取得できません。")
                        except Exception as e:
                            st.warning(f"特徴量重要度のプロットに失敗しました: {str(e)}")
            
                with col2:
                    st.write("SHAP値による特徴量重要度")
                    with st.spinner('SHAP値を計算中...'):
                        try:
                            # SHAP値の計算
                            explainer = shap.Explainer(model, X_train_transformed)
                            shap_values = explainer(X_train_transformed)
                            # SHAPサマリープロット
                            shap.summary_plot(shap_values, X_train_transformed, plot_type="bar", show=False)
                            st.pyplot(plt.gcf())
                            # キャプションを追加
                            st.caption("**SHAP値に基づく特徴量の重要度を示しています。各特徴量が予測結果に与える影響を定量的に評価できます。（青：正の影響　緑：負の影響）**")
                            plt.clf()
                            # SHAP値をデータフレームで表示
                            with st.expander("SHAP値データ", expanded=False):
                                shap_df = pd.DataFrame({
                                    '特徴量': feature_names,
                                    'SHAP値の平均絶対値': np.abs(shap_values.values).mean(axis=0)
                                }).sort_values(by='SHAP値の平均絶対値', ascending=False)
                                st.dataframe(shap_df, use_container_width=True)
                        except Exception as e:
                            st.warning(f"SHAP値の計算中にエラーが発生しました: {str(e)}")
            
            except Exception as e:
                st.error(f"特徴量重要度の分析中にエラーが発生しました: {str(e)}")
            st.markdown("---")
        
            # 2. 予測性能の分析
            st.subheader("3-2. 予測性能の分析と外れ値の検出")
            try:
                with st.spinner('予測性能を分析中...'):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("残差プロット")
                        plot_model(model, plot='residuals', display_format="streamlit")
                        st.caption("残差（予測値と実測値の差）と予測値の関係を示しています。パターンがなければ、モデルが適切にフィットしていることを示唆します。")
            
                    with col2:
                        st.write("予測誤差プロット")
                        plot_model(model, plot='error', display_format="streamlit")
                        st.caption("予測値と実測値の差を示しています。誤差が小さいほど、モデルの予測精度が高いことを示します。")
            
                    # 追加の評価図を表示
                    col3, col4 = st.columns(2)
                    with col3:
                        st.write("学習曲線")
                        plot_model(model, plot='learning', display_format="streamlit")
                        st.caption("トレーニングデータと検証データに対するモデルのパフォーマンスを示します。学習曲線が収束していれば、モデルが適切に学習していることを示します。")
            
                    with col4:
                        st.write("クックの距離")
                        plot_model(model, plot='cooks', display_format="streamlit")
                        st.caption("クックの距離が大きいデータポイントは、**モデルに強い影響を与える可能性がある**ため、外れ値の検出に役立ちます。")
            except Exception as e:
                st.error(f"予測性能の分析中にエラーが発生しました: {str(e)}")
        
            # 4. 決定木の可視化（該当モデルの場合のみ）
            if selected_model in ['dt', 'rf', 'et', 'gbr', 'xgboost', 'lightgbm']:
                st.markdown("---")
                st.subheader("3-4. 決定木の構造")
                try:
                    with st.spinner('決定木を可視化中...'):
                        if selected_model == 'dt':
                            # 決定木モデルの場合、直接ツリーをプロット
                            # 図を描画
                            fig, ax = plt.subplots(figsize=(40, 20))
                            plot_tree(
                                model,
                                feature_names=feature_names,
                                filled=True,
                                rounded=True,
                                fontsize=12,
                                ax=ax
                            )
                            st.pyplot(fig)
                            st.caption("決定木の構造を表示しています。")
                        else:
                            # ランダムフォレストや勾配ブースティングなどの場合
                            from sklearn.metrics import mean_squared_error
            
                            # テストデータを取得
                            X_test_transformed = get_config('X_test_transformed')
                            y_test = get_config('y_test')
            
                            # 各決定木の性能を評価
                            if selected_model in ['rf', 'et']:
                                estimators = model.estimators_
                            elif selected_model == 'gbr':
                                estimators = [est[0] for est in model.estimators_]
                            elif selected_model == 'xgboost':
                                import xgboost as xgb
                                estimators = model.get_booster().get_dump()
                            elif selected_model == 'lightgbm':
                                import lightgbm as lgb
                                estimators = model.booster_.dump_model()['tree_info']
                            else:
                                estimators = []
            
                            best_score = float('inf')
                            best_estimator_index = 0
            
                            for idx, estimator in enumerate(estimators):
                                if selected_model in ['rf', 'et', 'gbr']:
                                    y_pred = estimator.predict(X_test_transformed)
                                elif selected_model == 'xgboost':
                                    y_pred = model.predict(X_test_transformed, ntree_limit=idx+1)
                                elif selected_model == 'lightgbm':
                                    y_pred = model.predict(X_test_transformed, num_iteration=idx+1)
                                else:
                                    continue
            
                                mse = mean_squared_error(y_test, y_pred)
                                if mse < best_score:
                                    best_score = mse
                                    best_estimator_index = idx
            
                            # ベストなツリーを取得
                            if selected_model in ['rf', 'et', 'gbr']:
                                best_tree = estimators[best_estimator_index]
                                # 図を描画
                                fig, ax = plt.subplots(figsize=(40, 20))
                                plot_tree(
                                    best_tree,
                                    feature_names=feature_names,
                                    filled=True,
                                    rounded=True,
                                    fontsize=12,
                                    ax=ax
                                )
                                st.pyplot(fig)
                                st.caption(f"ベストな決定木（ツリー番号: {best_estimator_index}）の構造を表示しています。")
                            elif selected_model == 'xgboost':
                                # xgboostの場合
                                import xgboost as xgb
                                booster = model.get_booster()
                                fig, ax = plt.subplots(figsize=(40, 20))
                                xgb.plot_tree(booster, num_trees=best_estimator_index, ax=ax)
                                st.pyplot(fig)
                                st.caption(f"ベストな決定木（ツリー番号: {best_estimator_index}）の構造を表示しています。")
                            elif selected_model == 'lightgbm':
                                # lightgbmの場合
                                import lightgbm as lgb
                                graph = lgb.create_tree_digraph(model, tree_index=best_estimator_index)
                                st.graphviz_chart(graph)
                                st.caption(f"ベストな決定木（ツリー番号: {best_estimator_index}）の構造を表示しています。")
                            else:
                                st.warning(f"{selected_model}モデルの決定木の可視化は現在サポートされていません。")
                except Exception as e:
                    st.error(f"決定木の可視化中にエラーが発生しました: {str(e)}")

            # 5. モデルのファイナライズ
            st.markdown("---")
            st.header("4. モデルのファイナライズ")
            
            if st.button('モデルをファイナライズ', use_container_width=True):
                try:
                    with st.spinner("モデルをファイナライズ中..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        # モデルのファイナライズ
                        status_text.text("モデルをファイナライズ中...")
                        progress_bar.progress(30)
                        final_model = finalize_model(model)
                        
                        # 最終評価
                        status_text.text("最終評価を実行中...")
                        progress_bar.progress(60)
                        predictions = predict_model(final_model)
                        final_scores = pull()
                        
                        # 評価結果の表示
                        st.subheader("ファイナルモデルの評価結果")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("チューニング後の評価結果")
                            st.dataframe(st.session_state.model_configs['post_tuned_scores'], use_container_width=True)
                        with col2:
                            st.write("ファイナライズ後の評価結果")
                            st.dataframe(final_scores, use_container_width=True)


                        # 目的変数の名前をモデルに保存
                        final_model.target_column = target_variable

                        # モデルの保存
                        status_text.text("モデルを保存中...")
                        progress_bar.progress(90)
                        model_name = f"{target_variable}_finalized_model"

                        # モデルの保存
                        save_model(final_model, model_name)
                        
                        # モデルの読み込み
                        with open(f"{model_name}.pkl", 'rb') as f:
                            model_bytes = f.read()
                        
                        st.download_button(
                            label="ファイナライズしたモデルをダウンロード",
                            data=model_bytes,
                            file_name=f"{model_name}.pkl",
                            mime="application/octet-stream",
                            use_container_width=True
                        )
                        
                        progress_bar.progress(100)
                        status_text.text("ファイナライズが完了しました！")
                        st.success("✅ モデルのファイナライズが完了しました！")

                except Exception as e:
                    st.error(f"モデルのファイナライズ中にエラーが発生しました: {str(e)}")

    except Exception as e:
        st.error(f"予期せぬエラーが発生しました: {str(e)}")

# コピーライト情報
st.markdown("---")
st.caption('© 2022-2024 Dit-Lab.(Daiki Ito). All Rights Reserved.')
