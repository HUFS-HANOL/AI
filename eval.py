import openai
import json
import time

# OpenAI API 키 설정
openai.api_key = "" #openapi key 일단 가림

# 감정 카테고리
emotions = ["기쁨", "슬픔", "화남", "지침", "신남", "평온"]

# LLM 프롬프트 템플릿
def build_prompt(poem_content):
    return f"""
다음은 한국어 시입니다.
당신은 이 시를 읽고 느껴지는 감정을 하나의 단어로 표현하세요.
반드시 아래 감정 중 하나만 답하세요:

1) 기쁨
2) 슬픔
3) 화남
4) 지침
5) 신남
6) 평온

시:
{poem_content}

감정:"""

# LLM 쿼리
def query_emotion(poem_content):
    prompt = build_prompt(poem_content)
    
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    
    # 응답 파싱
    result = response["choices"][0]["message"]["content"].strip()
    # 만약에 앞에 "감정:" 같이 나올 경우 정리
    result = result.replace("감정:", "").strip()
    return result

# 평가 실행
def evaluate(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    correct = 0
    total = len(data)
    detailed_results = []

    print(f"총 {total}개 시 평가 시작...\n")

    for entry in data:
        poem_id = entry["id"]
        poem_content = entry["content"]
        target_emotion = entry["golden_emotion"]
        
        predicted_emotion = query_emotion(poem_content)
        
        result_line = f"[ID: {poem_id}] Target: {target_emotion} | Predicted: {predicted_emotion}"
        print(result_line)
        
        detailed_results.append({
            "id": poem_id,
            "target_emotion": target_emotion,
            "predicted_emotion": predicted_emotion
        })

        if predicted_emotion == target_emotion:
            correct += 1
        
        # 쿼리 간 sleep 넣기 (rate limit 고려)
        time.sleep(1)

    accuracy = correct / total
    print(f"\n최종 Accuracy: {accuracy:.2f}")

    # detailed results 저장 (optional, 원하면 파일로 저장 가능)
    with open("evaluation_results.json", "w", encoding="utf-8") as out_f:
        json.dump(detailed_results, out_f, indent=2, ensure_ascii=False)

# 메인 실행
if __name__ == "__main__":
    evaluate("/Users/suyunkim/projects/capstone/data/generated_poems_ex.json")
