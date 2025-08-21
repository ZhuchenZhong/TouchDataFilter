/*
 * 触摸数据结构定义
 * 用于处理和存储触摸屏幕的原始数据
 */

#pragma once

#include <cstdint>


typedef struct {
    int32_t* data;      // 指向触摸数据的指针，使用有符号整数
    uint32_t size;      // 触摸数据的大小
    uint32_t txn;       // TX 数目
    uint32_t rxn;       // RX 数目
    uint32_t timestamp; // 时间戳
    uint32_t frame_id;  // 帧 ID
} TouchFrame;

// 触摸帧集合，双向链表实现
typedef struct TouchFrameNode {
    TouchFrame frame;              // 当前触摸帧
    struct TouchFrameNode* prev;   // 前一个节点
    struct TouchFrameNode* next;   // 后一个节点
} TouchFrameNode;

typedef struct {
    TouchFrameNode* head;          // 链表头节点
    TouchFrameNode* tail;          // 链表尾节点
    uint32_t count;                // 帧数量
} TouchFrames;
