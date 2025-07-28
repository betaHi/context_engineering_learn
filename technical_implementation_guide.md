# 上下文工程实践指南：技术实现与案例分析

## 1. 提示工程实践案例

### 1.1 数学推理任务优化

#### 基础版本
```
问题：如果一个班级有30名学生，其中60%是女生，那么男生有多少人？
```

#### 优化后的思维链版本
```
让我们一步步解决这个问题：

问题：如果一个班级有30名学生，其中60%是女生，那么男生有多少人？

解题步骤：
1. 首先计算女生人数
2. 然后计算男生人数
3. 验证答案的正确性

让我开始计算：
```

#### 进一步优化的自验证版本
```
作为一个数学老师，我需要仔细解决这个问题并检查我的答案。

问题：如果一个班级有30名学生，其中60%是女生，那么男生有多少人？

解题策略：
- 方法1：直接计算男生比例
- 方法2：先算女生人数再减法
- 最后用两种方法互相验证

详细计算过程：
方法1：男生比例 = 100% - 60% = 40%
男生人数 = 30 × 40% = 12人

方法2：女生人数 = 30 × 60% = 18人
男生人数 = 30 - 18 = 12人

验证：12 + 18 = 30 ✓

答案：男生有12人。
```

### 1.2 代码生成任务优化

#### 基础版本
```
写一个Python函数来计算斐波那契数列
```

#### 优化后的详细规范版本
```
请为我创建一个Python函数来计算斐波那契数列，具体要求如下：

功能要求：
- 函数名：fibonacci
- 参数：n (正整数，表示要计算的斐波那契数的位置)
- 返回值：第n个斐波那契数
- 边界条件：n=1时返回0，n=2时返回1

性能要求：
- 使用动态规划方法，时间复杂度O(n)
- 包含输入验证
- 添加详细的文档字符串

代码风格：
- 遵循PEP 8规范
- 使用类型注释
- 包含使用示例

测试要求：
- 包含至少3个测试用例
- 覆盖边界情况和常规情况

请生成完整的函数实现：
```

### 1.3 创意写作任务优化

#### 基础版本
```
写一个关于人工智能的故事
```

#### 优化后的结构化版本
```
请创作一个科幻短篇小说，具体设定如下：

故事背景：
- 时间：2045年
- 地点：一个高度自动化的智慧城市
- 主题：人工智能与人类的协作关系

角色设定：
- 主角：一名AI研究员，对技术充满热情但内心孤独
- 配角：一个具有情感模拟能力的AI助手
- 冲突：AI助手开始表现出超越程序设定的"真实"情感

情节要求：
- 开头：建立主角的日常生活和研究环境
- 发展：AI助手的"异常"行为逐渐显现
- 高潮：主角必须决定是否相信AI的情感
- 结尾：留下开放性思考

写作风格：
- 第三人称叙述
- 现实主义与科幻元素结合
- 长度：800-1200字
- 语调：深思而富有同理心

请开始创作：
```

## 2. RAG系统实现详解

### 2.1 检索器设计模式

#### 密集检索实现
```python
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

class DenseRetriever:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.encoder = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        self.embeddings = None
    
    def build_index(self, documents):
        """构建文档索引"""
        self.documents = documents
        # 编码所有文档
        self.embeddings = self.encoder.encode(documents)
        
        # 创建FAISS索引
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings.astype('float32'))
    
    def retrieve(self, query, top_k=5):
        """检索相关文档"""
        query_embedding = self.encoder.encode([query])
        scores, indices = self.index.search(
            query_embedding.astype('float32'), top_k
        )
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            results.append({
                'document': self.documents[idx],
                'score': float(score),
                'rank': i + 1
            })
        
        return results
```

#### 混合检索策略
```python
class HybridRetriever:
    def __init__(self, dense_weight=0.7, sparse_weight=0.3):
        self.dense_retriever = DenseRetriever()
        self.sparse_retriever = BM25Retriever()  # 假设已实现
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
    
    def retrieve(self, query, top_k=10):
        """混合检索策略"""
        # 密集检索结果
        dense_results = self.dense_retriever.retrieve(query, top_k * 2)
        # 稀疏检索结果
        sparse_results = self.sparse_retriever.retrieve(query, top_k * 2)
        
        # 分数归一化和融合
        combined_scores = {}
        
        # 处理密集检索结果
        for result in dense_results:
            doc_id = result['document']
            normalized_score = 1 / (1 + result['score'])  # 距离转相似度
            combined_scores[doc_id] = self.dense_weight * normalized_score
        
        # 处理稀疏检索结果
        for result in sparse_results:
            doc_id = result['document']
            if doc_id in combined_scores:
                combined_scores[doc_id] += self.sparse_weight * result['score']
            else:
                combined_scores[doc_id] = self.sparse_weight * result['score']
        
        # 排序并返回top_k结果
        sorted_results = sorted(
            combined_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_k]
        
        return [{'document': doc, 'score': score} for doc, score in sorted_results]
```

### 2.2 上下文压缩技术

#### 基于重要性的压缩
```python
class ContextCompressor:
    def __init__(self, max_tokens=2048):
        self.max_tokens = max_tokens
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    
    def compress_by_importance(self, contexts, query):
        """基于重要性分数压缩上下文"""
        # 计算每个上下文片段的重要性
        importance_scores = []
        for context in contexts:
            score = self._calculate_importance(context, query)
            importance_scores.append(score)
        
        # 按重要性排序
        ranked_contexts = sorted(
            zip(contexts, importance_scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        # 贪心选择，直到达到token限制
        selected_contexts = []
        current_tokens = 0
        
        for context, score in ranked_contexts:
            context_tokens = len(self.tokenizer.encode(context))
            if current_tokens + context_tokens <= self.max_tokens:
                selected_contexts.append(context)
                current_tokens += context_tokens
            else:
                # 尝试截断当前context以填满剩余空间
                remaining_tokens = self.max_tokens - current_tokens
                if remaining_tokens > 50:  # 最小有意义长度
                    truncated = self._truncate_context(context, remaining_tokens)
                    selected_contexts.append(truncated)
                break
        
        return selected_contexts
    
    def _calculate_importance(self, context, query):
        """计算上下文重要性分数"""
        # 简化版本：基于查询词重叠度
        query_words = set(query.lower().split())
        context_words = set(context.lower().split())
        overlap = len(query_words.intersection(context_words))
        return overlap / len(query_words) if query_words else 0
```

### 2.3 迭代检索系统

```python
class IterativeRAG:
    def __init__(self, retriever, generator, max_iterations=3):
        self.retriever = retriever
        self.generator = generator
        self.max_iterations = max_iterations
    
    def generate_with_iteration(self, query):
        """迭代检索生成"""
        contexts = []
        current_query = query
        
        for iteration in range(self.max_iterations):
            # 检索相关文档
            retrieved_docs = self.retriever.retrieve(current_query, top_k=5)
            
            # 添加到上下文中
            for doc in retrieved_docs:
                if doc['document'] not in [c['content'] for c in contexts]:
                    contexts.append({
                        'content': doc['document'],
                        'score': doc['score'],
                        'iteration': iteration
                    })
            
            # 生成中间答案
            intermediate_answer = self._generate_intermediate(
                current_query, contexts
            )
            
            # 检查是否需要更多信息
            if self._is_sufficient(intermediate_answer, query):
                break
            
            # 生成下一轮查询
            current_query = self._generate_followup_query(
                query, intermediate_answer, contexts
            )
        
        # 生成最终答案
        final_answer = self._generate_final(query, contexts)
        return final_answer, contexts
    
    def _is_sufficient(self, answer, original_query):
        """判断答案是否充分"""
        # 简化版本：检查答案长度和关键词覆盖
        if len(answer.split()) < 20:
            return False
        
        query_keywords = set(original_query.lower().split())
        answer_keywords = set(answer.lower().split())
        coverage = len(query_keywords.intersection(answer_keywords)) / len(query_keywords)
        
        return coverage > 0.6
```

## 3. 多模态上下文处理

### 3.1 视觉-语言上下文整合

```python
class MultimodalContextProcessor:
    def __init__(self):
        self.vision_model = CLIP()  # 假设已加载
        self.text_model = SentenceTransformer()
    
    def process_image_text_context(self, image, text_query):
        """处理图像-文本上下文"""
        # 图像特征提取
        image_features = self.vision_model.encode_image(image)
        
        # 生成图像描述
        image_description = self._generate_image_description(image)
        
        # 构建多模态上下文
        multimodal_context = {
            'visual_features': image_features,
            'image_description': image_description,
            'text_query': text_query,
            'combined_prompt': self._create_combined_prompt(
                image_description, text_query
            )
        }
        
        return multimodal_context
    
    def _create_combined_prompt(self, image_desc, text_query):
        """创建组合提示"""
        return f"""
根据以下图像信息回答问题：

图像描述：{image_desc}

问题：{text_query}

请结合图像信息给出详细回答：
"""
```

### 3.2 跨模态对齐优化

```python
class CrossModalAligner:
    def __init__(self):
        self.alignment_model = SentenceTransformer('clip-ViT-B-32')
    
    def align_modalities(self, text_inputs, visual_inputs):
        """对齐不同模态的输入"""
        # 编码文本和视觉输入
        text_embeddings = self.alignment_model.encode(text_inputs)
        visual_embeddings = self.alignment_model.encode(visual_inputs)
        
        # 计算相似度矩阵
        similarity_matrix = np.dot(text_embeddings, visual_embeddings.T)
        
        # 找到最佳对齐
        alignments = []
        for i, text_emb in enumerate(text_embeddings):
            best_match_idx = np.argmax(similarity_matrix[i])
            alignments.append({
                'text_idx': i,
                'visual_idx': best_match_idx,
                'similarity': similarity_matrix[i][best_match_idx]
            })
        
        return alignments
```

## 4. 评估与监控框架

### 4.1 自动化评估系统

```python
class ContextEvaluator:
    def __init__(self):
        self.bleu_scorer = BLEU()
        self.rouge_scorer = Rouge()
        self.semantic_similarity = SentenceTransformer()
    
    def comprehensive_evaluate(self, predictions, references, contexts):
        """综合评估上下文工程效果"""
        metrics = {}
        
        # 文本质量指标
        metrics['bleu'] = self._compute_bleu(predictions, references)
        metrics['rouge'] = self._compute_rouge(predictions, references)
        metrics['semantic_similarity'] = self._compute_semantic_similarity(
            predictions, references
        )
        
        # 上下文利用指标
        metrics['context_utilization'] = self._compute_context_utilization(
            predictions, contexts
        )
        metrics['context_relevance'] = self._compute_context_relevance(
            contexts, references
        )
        
        # 效率指标
        metrics['response_time'] = self._measure_response_time()
        metrics['token_efficiency'] = self._compute_token_efficiency(contexts)
        
        return metrics
    
    def _compute_context_utilization(self, predictions, contexts):
        """计算上下文利用率"""
        utilization_scores = []
        
        for pred, ctx_list in zip(predictions, contexts):
            pred_words = set(pred.lower().split())
            
            total_context_words = set()
            for ctx in ctx_list:
                total_context_words.update(ctx.lower().split())
            
            if total_context_words:
                utilization = len(pred_words.intersection(total_context_words)) / len(total_context_words)
                utilization_scores.append(utilization)
        
        return np.mean(utilization_scores) if utilization_scores else 0
```

### 4.2 实时监控系统

```python
class ContextMonitor:
    def __init__(self, alert_threshold=0.1):
        self.performance_history = []
        self.alert_threshold = alert_threshold
        self.current_session = {}
    
    def log_performance(self, query, response, metrics, timestamp=None):
        """记录性能数据"""
        if timestamp is None:
            timestamp = datetime.now()
        
        performance_record = {
            'timestamp': timestamp,
            'query': query,
            'response': response,
            'metrics': metrics,
            'session_id': self.current_session.get('id')
        }
        
        self.performance_history.append(performance_record)
        self._check_performance_drift(metrics)
    
    def _check_performance_drift(self, current_metrics):
        """检测性能漂移"""
        if len(self.performance_history) < 10:
            return
        
        recent_metrics = [
            record['metrics'] for record in self.performance_history[-10:]
        ]
        historical_avg = np.mean([
            record['metrics']['semantic_similarity'] 
            for record in self.performance_history[:-10]
        ])
        
        recent_avg = np.mean([
            metrics['semantic_similarity'] for metrics in recent_metrics
        ])
        
        drift = abs(historical_avg - recent_avg) / historical_avg
        
        if drift > self.alert_threshold:
            self._trigger_alert(drift, historical_avg, recent_avg)
    
    def generate_performance_report(self, time_range='24h'):
        """生成性能报告"""
        end_time = datetime.now()
        if time_range == '24h':
            start_time = end_time - timedelta(hours=24)
        elif time_range == '7d':
            start_time = end_time - timedelta(days=7)
        else:
            start_time = end_time - timedelta(hours=1)
        
        relevant_records = [
            record for record in self.performance_history
            if start_time <= record['timestamp'] <= end_time
        ]
        
        report = {
            'time_range': time_range,
            'total_queries': len(relevant_records),
            'avg_response_time': np.mean([
                record['metrics']['response_time'] 
                for record in relevant_records
            ]),
            'avg_quality_score': np.mean([
                record['metrics']['semantic_similarity']
                for record in relevant_records
            ]),
            'quality_trend': self._calculate_trend(relevant_records),
            'top_issues': self._identify_top_issues(relevant_records)
        }
        
        return report
```

## 5. 部署与优化策略

### 5.1 缓存优化

```python
class ContextCache:
    def __init__(self, max_size=1000, ttl=3600):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.ttl = ttl  # Time to live in seconds
    
    def get_cached_context(self, query_hash):
        """获取缓存的上下文"""
        current_time = time.time()
        
        if query_hash in self.cache:
            # 检查是否过期
            if current_time - self.access_times[query_hash] < self.ttl:
                self.access_times[query_hash] = current_time
                return self.cache[query_hash]
            else:
                # 清除过期缓存
                del self.cache[query_hash]
                del self.access_times[query_hash]
        
        return None
    
    def cache_context(self, query_hash, context):
        """缓存上下文"""
        current_time = time.time()
        
        # 如果缓存已满，移除最旧的项
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.access_times, key=self.access_times.get)
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[query_hash] = context
        self.access_times[query_hash] = current_time
```

### 5.2 负载均衡与扩展

```python
class ContextEngineCluster:
    def __init__(self, engines):
        self.engines = engines
        self.load_balancer = RoundRobinBalancer()
        self.health_checker = HealthChecker()
    
    def process_request(self, query, context_requirements):
        """处理请求并负载均衡"""
        # 选择可用的引擎
        available_engines = self.health_checker.get_healthy_engines(self.engines)
        
        if not available_engines:
            raise Exception("No healthy engines available")
        
        # 负载均衡选择
        selected_engine = self.load_balancer.select_engine(available_engines)
        
        # 处理请求
        try:
            result = selected_engine.process(query, context_requirements)
            self.load_balancer.report_success(selected_engine)
            return result
        except Exception as e:
            self.load_balancer.report_failure(selected_engine)
            # 故障转移到其他引擎
            return self._failover_process(query, context_requirements, available_engines, selected_engine)
```

这份详细的技术实现指南涵盖了上下文工程的各个重要方面，包括具体的代码实现、最佳实践和部署策略。每个部分都提供了可直接使用的代码示例和详细的解释，帮助您深入理解和应用上下文工程技术。
