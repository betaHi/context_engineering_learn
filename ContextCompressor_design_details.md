# ContextCompressor 设计详解

## 1. 设计动机

在实际的RAG系统中，我们经常遇到以下挑战：

### 问题描述
```python
# 典型场景：检索到大量相关文档
retrieved_docs = [
    "文档1：关于Python编程的基础知识...(1000个token)",
    "文档2：Python函数定义的详细说明...(800个token)", 
    "文档3：递归算法的实现原理...(1200个token)",
    "文档4：动态规划算法详解...(900个token)",
    "文档5：斐波那契数列的数学原理...(600个token)"
]

# 问题：总共4500个token，但模型限制是2048个token
# 如何智能地选择最重要的内容？
```

### 朴素解决方案的问题
```python
# ❌ 简单截断：信息丢失严重
def naive_truncate(docs, max_tokens):
    result = []
    current_tokens = 0
    for doc in docs:
        doc_tokens = len(doc.split())
        if current_tokens + doc_tokens <= max_tokens:
            result.append(doc)
            current_tokens += doc_tokens
        else:
            break
    return result

# ❌ 随机选择：无法保证质量
def random_select(docs, max_tokens):
    import random
    random.shuffle(docs)
    return naive_truncate(docs, max_tokens)
```

## 2. ContextCompressor的核心创新

### 2.1 多维度重要性评分
```python
def calculate_importance_score(self, context, query, all_contexts):
    """综合重要性评分算法"""
    
    # 1. 相关性评分 (Relevance Score)
    relevance = self._calculate_relevance(context, query)
    
    # 2. 信息密度评分 (Information Density)
    density = self._calculate_information_density(context)
    
    # 3. 新颖性评分 (Novelty Score)
    novelty = self._calculate_novelty(context, all_contexts)
    
    # 4. 位置权重 (Position Weight)
    position_weight = self._calculate_position_weight(context, all_contexts)
    
    # 综合评分
    importance = (
        0.4 * relevance +      # 与查询的相关性最重要
        0.3 * density +        # 信息密度次之
        0.2 * novelty +        # 新颖性防止冗余
        0.1 * position_weight  # 位置信息作为辅助
    )
    
    return importance
```

### 2.2 智能截断策略
```python
def intelligent_truncate(self, context, max_tokens, query):
    """基于语义的智能截断"""
    
    # 将文档分解为句子
    sentences = self._split_into_sentences(context)
    
    # 计算每个句子的重要性
    sentence_scores = []
    for sentence in sentences:
        score = self._calculate_relevance(sentence, query)
        sentence_scores.append((sentence, score))
    
    # 按重要性排序
    sorted_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)
    
    # 贪心选择最重要的句子
    selected_sentences = []
    current_tokens = 0
    
    for sentence, score in sorted_sentences:
        sentence_tokens = len(sentence.split())
        if current_tokens + sentence_tokens <= max_tokens:
            selected_sentences.append(sentence)
            current_tokens += sentence_tokens
        else:
            # 如果剩余空间够大，尝试截断当前句子
            remaining_tokens = max_tokens - current_tokens
            if remaining_tokens > 20:  # 至少保留20个token的句子片段
                truncated = self._truncate_sentence(sentence, remaining_tokens)
                selected_sentences.append(truncated)
            break
    
    # 按原始顺序重新排列（保持逻辑连贯性）
    return self._reorder_sentences(selected_sentences, sentences)
```

## 3. 核心算法实现

### 3.1 相关性计算
```python
def _calculate_relevance(self, context, query):
    """计算上下文与查询的相关性"""
    
    # 方法1：词汇重叠度
    query_words = set(query.lower().split())
    context_words = set(context.lower().split())
    
    # Jaccard相似度
    jaccard = len(query_words.intersection(context_words)) / len(query_words.union(context_words))
    
    # 方法2：TF-IDF相似度
    tfidf_sim = self._calculate_tfidf_similarity(context, query)
    
    # 方法3：语义相似度（如果有embedding模型）
    if hasattr(self, 'embedding_model'):
        semantic_sim = self._calculate_semantic_similarity(context, query)
        return 0.3 * jaccard + 0.3 * tfidf_sim + 0.4 * semantic_sim
    else:
        return 0.5 * jaccard + 0.5 * tfidf_sim

def _calculate_tfidf_similarity(self, context, query):
    """使用TF-IDF计算相似度"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([context, query])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    return similarity
```

### 3.2 信息密度计算
```python
def _calculate_information_density(self, context):
    """计算文本的信息密度"""
    
    words = context.split()
    
    # 1. 词汇多样性 (Lexical Diversity)
    unique_words = len(set(words))
    total_words = len(words)
    lexical_diversity = unique_words / total_words if total_words > 0 else 0
    
    # 2. 实体密度 (Entity Density)
    # 简化版本：计算大写词汇比例（通常是专有名词）
    capital_words = sum(1 for word in words if word[0].isupper())
    entity_density = capital_words / total_words if total_words > 0 else 0
    
    # 3. 数字和符号密度（通常包含重要信息）
    import re
    numbers_symbols = len(re.findall(r'[\d\.\,\%\$\@\#]', context))
    symbol_density = numbers_symbols / len(context) if len(context) > 0 else 0
    
    # 综合密度分数
    density = 0.5 * lexical_diversity + 0.3 * entity_density + 0.2 * symbol_density
    
    return min(density, 1.0)  # 限制在[0,1]范围内
```

### 3.3 新颖性评估
```python
def _calculate_novelty(self, context, all_contexts):
    """计算文本的新颖性（避免冗余）"""
    
    if len(all_contexts) <= 1:
        return 1.0
    
    # 计算与其他上下文的最大相似度
    max_similarity = 0.0
    
    for other_context in all_contexts:
        if other_context == context:
            continue
            
        # 使用简单的词汇重叠度计算相似度
        words1 = set(context.lower().split())
        words2 = set(other_context.lower().split())
        
        if len(words1.union(words2)) > 0:
            similarity = len(words1.intersection(words2)) / len(words1.union(words2))
            max_similarity = max(max_similarity, similarity)
    
    # 新颖性 = 1 - 最大相似度
    novelty = 1.0 - max_similarity
    
    return novelty
```

## 4. 高级优化技术

### 4.1 动态阈值调整
```python
def _adaptive_threshold(self, contexts, query):
    """根据查询复杂度动态调整选择阈值"""
    
    # 分析查询复杂度
    query_complexity = self._analyze_query_complexity(query)
    
    if query_complexity > 0.8:  # 复杂查询需要更多上下文
        return 0.3  # 降低阈值，选择更多内容
    elif query_complexity < 0.3:  # 简单查询可以更严格
        return 0.7  # 提高阈值，选择最相关的内容
    else:
        return 0.5  # 默认阈值

def _analyze_query_complexity(self, query):
    """分析查询复杂度"""
    
    factors = []
    
    # 1. 查询长度
    word_count = len(query.split())
    length_factor = min(word_count / 20, 1.0)  # 标准化到[0,1]
    
    # 2. 专业词汇密度
    technical_words = ['算法', '函数', '实现', '优化', '性能', '复杂度']
    tech_count = sum(1 for word in technical_words if word in query)
    tech_factor = min(tech_count / 3, 1.0)
    
    # 3. 疑问词数量
    question_words = ['什么', '如何', '为什么', '怎样', '哪个', '怎么']
    question_count = sum(1 for word in question_words if word in query)
    question_factor = min(question_count / 2, 1.0)
    
    # 综合复杂度
    complexity = 0.4 * length_factor + 0.4 * tech_factor + 0.2 * question_factor
    
    return complexity
```

## 5. 性能优化策略

### 5.1 缓存机制
```python
class ContextCompressor:
    def __init__(self, max_tokens=2048, cache_size=1000):
        self.max_tokens = max_tokens
        self.relevance_cache = {}  # 缓存相关性计算结果
        self.embedding_cache = {}  # 缓存embedding计算结果
        self.cache_size = cache_size
    
    def _get_cached_relevance(self, context, query):
        """获取缓存的相关性分数"""
        cache_key = hash(context + query)
        
        if cache_key in self.relevance_cache:
            return self.relevance_cache[cache_key]
        
        # 计算并缓存
        relevance = self._calculate_relevance(context, query)
        
        # 限制缓存大小
        if len(self.relevance_cache) >= self.cache_size:
            # 移除最旧的条目
            oldest_key = next(iter(self.relevance_cache))
            del self.relevance_cache[oldest_key]
        
        self.relevance_cache[cache_key] = relevance
        return relevance
```

### 5.2 并行处理
```python
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

def parallel_compress(self, contexts, query, num_workers=4):
    """并行处理多个上下文片段"""
    
    # 将上下文分块
    chunk_size = len(contexts) // num_workers
    chunks = [contexts[i:i+chunk_size] for i in range(0, len(contexts), chunk_size)]
    
    # 并行计算重要性分数
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_chunk = {
            executor.submit(self._process_chunk, chunk, query): chunk 
            for chunk in chunks
        }
        
        all_scored_contexts = []
        for future in future_to_chunk:
            scored_contexts = future.result()
            all_scored_contexts.extend(scored_contexts)
    
    # 合并结果并最终选择
    return self._final_selection(all_scored_contexts, query)
```

## 6. 实际应用示例

```python
# 使用示例
compressor = ContextCompressor(max_tokens=2048)

# 原始上下文（过长）
long_contexts = [
    "Python是一种高级编程语言...(很长的文档)",
    "斐波那契数列是数学中的一个重要概念...(很长的文档)",
    "动态规划是解决斐波那契问题的最优方法...(很长的文档)"
]

# 用户查询
user_query = "如何用Python实现斐波那契数列的动态规划算法？"

# 智能压缩
compressed_contexts = compressor.compress_by_importance(long_contexts, user_query)

print(f"原始token数: {sum(len(ctx.split()) for ctx in long_contexts)}")
print(f"压缩后token数: {sum(len(ctx.split()) for ctx in compressed_contexts)}")
print(f"压缩率: {(1 - len(compressed_contexts)/len(long_contexts)) * 100:.1f}%")
```

这就是`ContextCompressor`的完整设计思路和实现原理！它不是简单的截断，而是基于多个维度的智能评估和选择。
