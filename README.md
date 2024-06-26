# Project
LLM을 이용한 개인 지식데이터베이스인 RAG기술중 Meta에서 만든 엄청 효율적이고 뛰어난 RAPTOR(Recursive Abstractive Processing for Tree-Organized)라는 논문을 llama index의 내부 ChromaDB를 통해 구현한 예제입니다.
https://arxiv.org/html/2401.18059v1

# RAPTOR Overview
방법자체는 이해하기 쉽게 말하자면 Low레벨은 기본적으로 원본그대로 내용을 가지고 있고 high레벨로 갈수록 low 레벨들의 클러스트로 묶은 요약본을 찾는 형태입니다. 여기서 각 묶음 단위는 chunk사이즈로 생각하면 됩니다.
예를들어 해리포터 1~10권을 RAPTOR로 구성한다라면 0레벨 원본 chunk 10개, 1레벨 총 요약본 chunk1개 입니다.
그래서 사용자가 만약 '해리포터 불사조기사단'을 찾는다 하면 먼저 1레벨 요약본에서 정보를 찾고, 0레벨 원본 책의 정보를 search해주는 방식입니다.

# Final Thoughts
개인적으로 제가 구현하고 있던 플랫폼의 도서관 방식과 비슷해서 감명 깊었고, 실제 몇번 해보니 정확도도 더 높아져서 나중에 본격적으로 RAG 구성시 참고해볼만한 방식입니다.
