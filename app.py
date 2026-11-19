import re  # 문자열 패턴 추출과 정리에 사용하는 정규표현식 모듈
import pandas as pd  # 표 데이터를 다루기 위한 판다스 라이브러리
import requests  # 웹 페이지 HTML을 가져오기 위한 라이브러리
import streamlit as st  # 웹 앱 화면을 만들기 위한 스트림릿 라이브러리
from bs4 import BeautifulSoup  # HTML 태그를 파싱하기 위한 뷰티풀수프 모듈


DEFAULT_URL = "https://ko.wikipedia.org/wiki/대한민국의_영화_흥행_기록"


def fetch_html(url: str) -> str:
    """주어진 주소에서 HTML 문자열을 가져오고 네트워크/HTTP 오류를 처리한다."""
    # 위키백과 요청 시 수업실습용임을 밝히는 User-Agent 헤더 설정
    headers = {
        "User-Agent": "PythonClassPracticeBot/1.0 (수업실습용; student-practice@example.com)"
    }
    try:
        # timeout을 지정해 응답이 지나치게 길어지는 것을 방지
        response = requests.get(url, headers=headers, timeout=10)
        # 4xx, 5xx 상태 코드가 나오면 HTTPError 발생
        response.raise_for_status()
        # 한글 깨짐 방지를 위해 인코딩 지정
        response.encoding = response.apparent_encoding or "utf-8"
        return response.text
    except requests.exceptions.Timeout as exc:
        raise RuntimeError("요청 시간이 초과되었습니다. 네트워크 상태를 확인하거나 잠시 후 다시 시도해 주세요.") from exc
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError("서버에 연결할 수 없습니다. 인터넷 연결 또는 주소를 확인해 주세요.") from exc
    except requests.exceptions.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else "알 수 없음"
        raise RuntimeError(f"HTTP 오류가 발생했습니다. (상태 코드: {status_code})") from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"요청 처리 중 오류가 발생했습니다: {exc}") from exc


def extract_target_table(html: str):
    """wikitable 중 첫 줄에 '관객수'와 '감독'이 모두 포함된 첫 번째 표를 찾는다."""
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table", class_="wikitable")

    for table in tables:
        rows = table.find_all("tr")
        if not rows:
            continue
        # 첫 번째 줄의 th 또는 td 텍스트를 수집해 조건 확인
        header_cells = rows[0].find_all(["th", "td"])
        header_texts = [cell.get_text(" ", strip=True) for cell in header_cells]
        has_audience = any("관객수" in text for text in header_texts)
        has_director = any("감독" in text for text in header_texts)
        if has_audience and has_director:
            # 표 내부의 각주 번호(sup 태그)를 미리 제거
            for sup in table.find_all("sup"):
                sup.decompose()
            return table

    raise ValueError("첫 줄에 '관객수'와 '감독'이 모두 있는 wikitable을 찾지 못했습니다.")


def parse_table_to_dataframe(table) -> pd.DataFrame:
    """HTML 표의 제목(th)과 데이터 줄(tr > td)을 읽어 DataFrame으로 변환한다."""
    rows = table.find_all("tr")

    # 첫 줄에서 열 이름 추출
    header_cells = rows[0].find_all(["th", "td"])
    headers = [cell.get_text(" ", strip=True) for cell in header_cells]

    # 중복 열 이름이 있을 경우 구분할 수 있도록 번호 부여
    unique_headers = []
    seen = {}
    for col in headers:
        base_name = col if col else "열"
        count = seen.get(base_name, 0) + 1
        seen[base_name] = count
        unique_headers.append(base_name if count == 1 else f"{base_name}_{count}")

    records = []
    for row in rows[1:]:
        cells = row.find_all("td")
        if not cells:
            continue
        values = [cell.get_text(" ", strip=True) for cell in cells]

        # 열 개수 불일치에 대비해 길이 맞춤
        if len(values) < len(unique_headers):
            values.extend([""] * (len(unique_headers) - len(values)))
        elif len(values) > len(unique_headers):
            values = values[: len(unique_headers)]

        records.append(dict(zip(unique_headers, values)))

    if not records:
        raise ValueError("표에서 영화 데이터 행을 찾지 못했습니다.")

    return pd.DataFrame(records)


def find_column_name(df: pd.DataFrame, keywords: list[str]) -> str | None:
    """키워드가 포함된 실제 열 이름을 찾아 반환한다."""
    for col in df.columns:
        if any(kw in col for kw in keywords):
            return col
    return None


def clean_movie_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """제목의 《》 제거, 관객수 숫자 변환, 개봉연도·개봉월 생성을 수행한다."""
    cleaned = df.copy()

    title_col = find_column_name(cleaned, ["영화", "제목"])
    audience_col = find_column_name(cleaned, ["관객수"])
    release_col = find_column_name(cleaned, ["개봉일", "개봉"])

    # 제목 정제: 《 와 》 기호 제거
    if title_col:
        cleaned["제목"] = (
            cleaned[title_col]
            .astype(str)
            .str.replace("《", "", regex=False)
            .str.replace("》", "", regex=False)
            .str.strip()
        )
    else:
        cleaned["제목"] = cleaned.iloc[:, 0].astype(str).str.strip()

    # 관객수 정제: 쉼표 등 제거 후 숫자로 변환
    if audience_col:
        cleaned["관객수_숫자"] = (
            cleaned[audience_col]
            .astype(str)
            .apply(lambda x: re.sub(r"[^0-9]", "", x))
        )
        cleaned["관객수_숫자"] = pd.to_numeric(cleaned["관객수_숫자"], errors="coerce")
    else:
        raise ValueError("'관객수' 열을 찾을 수 없어 숫자 변환을 할 수 없습니다.")

    # 개봉일에서 개봉연도와 개봉월 추출
    if release_col:
        release_series = cleaned[release_col].astype(str)
        # 4자리 연도 추출
        years = release_series.str.extract(r"(\d{4})")[0]
        # '년' 뒤의 월 또는 'YYYY-MM' / 'YYYY. MM.' 패턴에서 월 추출
        months_kor = release_series.str.extract(r"\d{4}\s*년\s*(\d{1,2})\s*월")[0]
        months_sep = release_series.str.extract(r"\d{4}[-./]\s*(\d{1,2})")[0]
        months = months_kor.fillna(months_sep)

        cleaned["개봉연도"] = pd.to_numeric(years, errors="coerce").astype("Int64")
        cleaned["개봉월"] = pd.to_numeric(months, errors="coerce").astype("Int64")
    else:
        cleaned["개봉연도"] = pd.Series([pd.NA] * len(cleaned), dtype="Int64")
        cleaned["개봉월"] = pd.Series([pd.NA] * len(cleaned), dtype="Int64")

    # 관객수 기준 내림차순 정렬
    cleaned = cleaned.sort_values(by="관객수_숫자", ascending=False, na_position="last").reset_index(drop=True)
    return cleaned


def render_metrics_and_table(df: pd.DataFrame):
    """핵심 지표(천만영화수, 역대 1위, 1위 관객수)와 정제된 표를 출력한다."""
    ten_million_count = int((df["관객수_숫자"] >= 10_000_000).sum())
    top_movie = df.iloc[0]["제목"] if len(df) > 0 else "-"
    top_audience = df.iloc[0]["관객수_숫자"] if len(df) > 0 and pd.notna(df.iloc[0]["관객수_숫자"]) else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("천만영화수", f"{ten_million_count}편")
    col2.metric("역대 1위", str(top_movie))
    col3.metric("1위 관객수", f"{int(top_audience):,}명")

    st.subheader("수집 및 정제된 영화 표")
    st.dataframe(df, use_container_width=True)


def render_charts(df: pd.DataFrame):
    """관객수 상위 10편 가로 막대차트와 개봉연도별 편수 막대차트를 그린다."""
    st.subheader("관객수 상위 10편 (가로 막대차트)")
    top10 = df.dropna(subset=["관객수_숫자"]).head(10).copy()
    if not top10.empty:
        # 가로 막대차트에서 1위가 위에 오도록 역순 정렬 후 인덱스를 영화 제목으로 설정
        top10_chart = top10.iloc[::-1].set_index("제목")[["관객수_숫자"]]
        st.bar_chart(top10_chart, horizontal=True)
    else:
        st.info("관객수 차트를 그릴 데이터가 없습니다.")

    st.subheader("개봉연도별 영화 편수")
    year_valid = df.dropna(subset=["개봉연도"])
    if not year_valid.empty:
        year_counts = (
            year_valid.groupby("개봉연도")["제목"]
            .count()
            .sort_index()
            .rename("영화편수")
        )
        # 연도 문자열을 인덱스로 사용해 막대차트 표시
        year_counts.index = year_counts.index.astype(str)
        st.bar_chart(year_counts)
    else:
        st.info("개봉연도 데이터가 없어 연도별 편수 차트를 그릴 수 없습니다.")


def run_app():
    """스트림릿 앱의 전체 실행 흐름을 제어한다."""
    st.set_page_config(page_title="천만관객 한국영화 흥행 분석", layout="wide")
    st.title("천만관객 한국영화 흥행 기록 수집 및 분석")

    # ① 위키백과 주소 입력창 (기본값 설정)
    url = st.text_input("위키백과 문서 주소", value=DEFAULT_URL)

    # ② 버튼 클릭 시 데이터 수집, 정제, 시각화를 수행
    if st.button("데이터 수집 및 분석 실행"):
        if not url.strip():
            st.error("주소를 입력해 주세요.")
            return

        try:
            with st.spinner("위키백과에서 표 데이터를 가져오는 중입니다..."):
                html = fetch_html(url.strip())  # ② HTML 가져오기
                target_table = extract_target_table(html)  # ③ 대상 표 찾기 및 sup 제거
                raw_df = parse_table_to_dataframe(target_table)  # ③ DataFrame 생성
                cleaned_df = clean_movie_dataframe(raw_df)  # ④ 데이터 정제

            render_metrics_and_table(cleaned_df)  # ⑤ 지표 및 표 출력
            render_charts(cleaned_df)  # ⑤ 차트 2종 출력

        except Exception as exc:
            # 연결실패, 시간초과, HTTP 오류, 표 미발견 등 오류 메시지 안내
            st.error(f"처리 중 문제가 발생했습니다: {exc}")


if __name__ == "__main__":
    run_app()